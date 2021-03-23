#include "metric/static.h"
#include "slice/static.h"
#include "query.h"
#include "match/matcher_impl.h"
#include "metric/alignment.h"
#include "metric/factory.h"

SimilarityMatrixRef StaticEmbeddingSimilarityMatrixFactory::build_similarity_matrix(
	const std::vector<StaticEmbeddingRef> &p_embeddings) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const Needle needle(m_query);

	const auto matrix = std::make_shared<SimilarityMatrix>();

	const size_t vocab_size = vocab->size();
	const size_t needle_size = static_cast<size_t>(needle.size());
	matrix->m_similarity.resize({ssize_t(vocab_size), ssize_t(needle_size)});

	const auto &needle_tokens = needle.token_ids();
	py::list sources;
	py::array_t<size_t> indices{static_cast<py::ssize_t>(needle_size)};
	auto mutable_indices = indices.mutable_unchecked<1>();

	for (size_t j = 0; j < needle_size; j++) { // for each token in needle
		const auto t = needle_tokens[j];
		size_t t_rel;
		const auto &t_vectors = pick_vectors(p_embeddings, t, t_rel);
		sources.append(t_vectors);
		mutable_indices[j] = t_rel;
	}

	const auto py_embeddings = py::module_::import("vectorian.embeddings");
	const auto needle_vectors = py_embeddings.attr("StackedVectors")(sources, indices);

	size_t offset = 0;
	for (const auto &embedding : p_embeddings) {
		const auto &vectors = embedding->vectors();
		const size_t size = embedding->size();

		m_metric.vector_metric(
			vectors,
			needle_vectors,
			xt::strided_view(matrix->m_similarity, {xt::range(offset, offset + size), xt::all()}));

		PPK_ASSERT(offset + size <= vocab_size);

		offset += size;
	}
	PPK_ASSERT(offset == vocab_size);

	for (size_t j = 0; j < needle.size(); j++) { // for each token in needle

		// since the j-th needle token is a specific vocabulary token, we always
		// set that specific vocabulary token similarity to 1 (regardless of the
		// embedding distance).
		const auto k = needle_tokens[j];
		if (k >= 0) {
			matrix->m_similarity(k, j) = 1.0f;
		}
	}

	return matrix;
}

void StaticEmbeddingSimilarityMatrixFactory::compute_magnitudes(
	const std::vector<StaticEmbeddingRef> &p_embeddings,
	const SimilarityMatrixRef &p_matrix) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const Needle needle(m_query);

	p_matrix->m_magnitudes.resize({static_cast<ssize_t>(vocab->size())});
	size_t offset = 0;
	for (const auto &embedding : p_embeddings) {
		const auto &vectors = embedding->vectors();
		const size_t size = embedding->size();

		const auto magnitudes = vectors.attr("magnitudes").cast<xt::pytensor<float, 1>>();
		xt::strided_view(p_matrix->m_magnitudes, {xt::range(offset, offset + size)}) = magnitudes;

		offset += size;
	}
	PPK_ASSERT(offset == vocab->size());
}

SimilarityMatrixRef StaticEmbeddingSimilarityMatrixFactory::create(
	const DocumentRef&) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const auto embeddings = vocab->get_compiled_embeddings(m_embedding_index);

	const auto matrix = build_similarity_matrix(embeddings);

	//std::cout << "has debug hook " << p_query->debug_hook().has_value() << "\n";
	/*if (p_query->debug_hook().has_value()) {
		auto gen_rows = py::cpp_function([&] () {
			const auto &vocab = p_query->vocabulary();

			py::list row_tokens;
			const size_t n = vocab->size();
			for (size_t i = 0; i < n; i++) {
				row_tokens.append(vocab->id_to_token(i));
			}
			return row_tokens;
		});

		auto gen_columns = py::cpp_function([&] () {
			const auto &vocab = p_query->vocabulary();

			py::list col_tokens;
			for (const auto &t : *p_query->tokens()) {
				col_tokens.append(vocab->id_to_token(t.id));
			}
			return col_tokens;
		});

		py::dict data;
		data["matrix"] = m_similarity;
		data["rows"] = gen_rows;
		data["columns"] = gen_columns;

		(*p_query->debug_hook())("similarity_matrix", data);
	}*/

	if (m_matcher_factory->needs_magnitudes()) {
		compute_magnitudes(
			embeddings,
			matrix);
	}

	return matrix;

	//return std::make_shared<StaticEmbeddingMetric>(
	//	m_embeddings[0]->name(), matrix, matcher_factory);
}
