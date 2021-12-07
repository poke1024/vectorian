#include "metric/static.h"
#include "metric/contextual.h"
#include "slice/static.h"
#include "query.h"
#include "match/matcher_impl.h"
#include "metric/alignment.h"
#include "metric/factory.h"

SimilarityMatrixRef StaticEmbeddingSimilarityMatrixFactory::build_static_similarity_matrix(
	const std::vector<StaticEmbeddingRef> &p_embeddings) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const Needle needle(m_query);

	const auto matrix = std::make_shared<StaticSimilarityMatrix>();

	const size_t vocab_size = vocab->size();
	const ssize_t needle_size = needle.size();
	matrix->m_similarity.resize({ssize_t(vocab_size), needle_size});

	if (needle_size < 1) {
		return matrix;
	}

	const auto &needle_tokens = needle.token_ids();
	py::list sources;
	py::array_t<size_t> indices{needle_size};
	auto mutable_indices = indices.mutable_unchecked<1>();

	for (ssize_t j = 0; j < needle_size; j++) { // for each token in needle
		const auto t = needle_tokens[j];
		size_t t_rel;
		const auto &t_vectors = pick_vectors(p_embeddings, t, t_rel);
		sources.append(t_vectors);
		mutable_indices[j] = t_rel;
	}

	const auto py_embeddings = py::module_::import("vectorian.embedding.vectors");
	const auto t_vectors = py_embeddings.attr("StackedVectors")(sources, indices);

	size_t offset = 0;
	for (const auto &embedding : p_embeddings) {
		const py::object s_vectors = embedding->vectors();
		const size_t size = embedding->size();

		const py::object tfm_t_vectors = s_vectors.attr(PY_TRANSFORM)(t_vectors);
		m_metric.vector_metric(
			s_vectors,
			tfm_t_vectors,
			xt::strided_view(matrix->m_similarity, {xt::range(offset, offset + size), xt::all()}));

		PPK_ASSERT(offset + size <= vocab_size);

		offset += size;
	}
	PPK_ASSERT(offset == vocab_size);

	for (ssize_t j = 0; j < needle_size; j++) { // for each token in needle

		// since the j-th needle token is a specific vocabulary token, we always
		// set that specific vocabulary token similarity to 1 (regardless of the
		// embedding distance).
		const auto k = needle_tokens[j];
		if (k >= 0) {
			matrix->m_similarity(k, j) = 1.0f;
		}
	}

	if (m_matcher_factory->needs_magnitudes()) {
		compute_magnitudes(
			p_embeddings,
			matrix);
	}

	matrix->clip();

	return matrix;
}

void StaticEmbeddingSimilarityMatrixFactory::compute_magnitudes(
	const std::vector<StaticEmbeddingRef> &p_embeddings,
	const SimilarityMatrixRef &p_matrix) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const Needle needle(m_query);

	p_matrix->m_magnitudes_s.resize({static_cast<ssize_t>(vocab->size())});
	size_t offset = 0;
	for (const auto &embedding : p_embeddings) {
		const py::object s_vectors = embedding->vectors();
		const size_t size = embedding->size();

		const auto magnitudes = s_vectors.attr(PY_MAGNITUDES).cast<xt::pytensor<float, 1>>();
		xt::strided_view(p_matrix->m_magnitudes_s, {xt::range(offset, offset + size)}) = magnitudes;

		offset += size;
	}
	PPK_ASSERT(offset == vocab->size());

	fill_magnitudes_t(p_matrix);
}

void StaticEmbeddingSimilarityMatrixFactory::fill_magnitudes_t(
	const SimilarityMatrixRef &p_matrix) {

	const Needle needle(m_query);
	PPK_ASSERT(p_matrix.get() != nullptr);
	const auto &static_mag = p_matrix->m_magnitudes_s;
	PPK_ASSERT(static_mag.shape(0) == m_query->vocabulary()->size());
	auto &mag_t = p_matrix->m_magnitudes_t;
	mag_t.resize({ssize_t(needle.size())});
	for (size_t i = 0; i < needle.size(); i++) {
		const auto t = needle.token_id(i);
		if (t >= 0) {
			mag_t(i) = static_mag(t);
		} else {
			mag_t(i) = 0.0f;
		}
	}
}

StaticEmbeddingSimilarityMatrixFactory::StaticEmbeddingSimilarityMatrixFactory(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const MatcherFactoryRef &p_matcher_factory,
	const size_t p_embedding_index) :

	m_query(p_query),
	m_metric(p_metric),
	m_matcher_factory(p_matcher_factory),
	m_embedding_index(p_embedding_index),

	PY_TRANSFORM("transform"),
	PY_MAGNITUDES("magnitudes") {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const auto embeddings = vocab->get_compiled_embeddings(m_embedding_index);

	m_static_matrix = build_static_similarity_matrix(embeddings);
}

SimilarityMatrixRef StaticEmbeddingSimilarityMatrixFactory::create(
	const EmbeddingType p_embedding_type,
	const DocumentRef &p_document) {

	const QueryVocabularyRef vocab = m_query->vocabulary();
	const auto embeddings = vocab->get_compiled_embeddings(m_embedding_index);

	switch (p_embedding_type) {
		case STATIC: {
			return m_static_matrix;
		} break;

		case CONTEXTUAL: {
			// we end up here if we use a static embedding in a contextual query,
			// i.e. if we are asked to fill a contextual similarity matrix of the
			// shape (n_tokens_in_doc, n_dims) to mix with similarities from other
			// contextual embeddings.

			const Needle needle(m_query);
			const auto matrix = std::make_shared<ContextualSimilarityMatrix>();

			const auto &static_sim = m_static_matrix->m_similarity;
			auto &contextual_sim = matrix->m_similarity;

			const size_t n_tokens = p_document->n_tokens();
			const auto &tokens = *p_document->tokens_vector();

			contextual_sim.resize({
				ssize_t(n_tokens), ssize_t(needle.size())});
			for (size_t i = 0; i < n_tokens; i++) {
				xt::strided_view(contextual_sim, {i, xt::all()}) =
					xt::strided_view(static_sim, {tokens[i].id, xt::all()});
			}

			if (m_matcher_factory->needs_magnitudes()) {
				const auto &static_mag = m_static_matrix->m_magnitudes_s;
				auto &contextual_mag_s = matrix->m_magnitudes_s;

				contextual_mag_s.resize({ssize_t(n_tokens)});
				for (size_t i = 0; i < n_tokens; i++) {
					contextual_mag_s(i) = static_mag(tokens[i].id);
				}

				fill_magnitudes_t(matrix);
			}

			return matrix;
		} break;

		default: {
			throw std::runtime_error("illegal embedding type");
		} break;
	}
}

void StaticSimilarityMatrix::call_hook(
	const QueryRef &p_query) const {

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
		for (const auto &t : *p_query->tokens_vector()) {
			col_tokens.append(vocab->id_to_token(t.id));
		}
		return col_tokens;
	});

	py::dict data;
	data["similarity"] = m_similarity;
	if (m_magnitudes_s.shape(0) > 0) {
		data["magnitudes_s"] = m_magnitudes_s;
		data["magnitudes_t"] = m_magnitudes_t;
	}
	data["rows"] = gen_rows;
	data["columns"] = gen_columns;

	(*p_query->debug_hook())("static_similarity_matrix", data);
}