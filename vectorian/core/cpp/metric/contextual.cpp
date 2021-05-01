#include "result_set.h"
#include "metric/contextual.h"
#include "metric/alignment.h"
#include "embedding/contextual.h"
#include "query.h"
#include "document.h"
#include "metric/factory.h"
#include "slice/contextual.h"

ContextualEmbeddingSimilarityMatrixFactory::ContextualEmbeddingSimilarityMatrixFactory(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const MatcherFactoryRef &p_matcher_factory,
	const size_t p_embedding_index) :

	m_query(p_query),
	m_metric(p_metric),
	m_matcher_factory(p_matcher_factory),
	m_embedding_index(p_embedding_index),

	PY_SIZE("size"),
	PY_MAGNITUDES("magnitudes"),
	PY_TRANSFORM("transform") {
}

SimilarityMatrixRef ContextualEmbeddingSimilarityMatrixFactory::create_with_py_context(
	const DocumentRef &p_document) {

	const auto embedding_name = m_query->vocabulary()->embedding_manager()->get_py_name(m_embedding_index);
	const auto &cache = m_query->vectors_cache();

	const HandleRef s_vectors = cache.open(
		p_document->get_contextual_embedding_vectors(embedding_name), p_document->n_tokens());
	const HandleRef t_vectors = cache.open(
		m_query->get_contextual_embedding_vectors(embedding_name), m_query->n_tokens());

	// compute a n x m matrix, (n: number of tokens in document, m: number of tokens in needle)
	// might offload this to GPU. use this as basis for ContextualEmbeddingSlice.

	const auto sim_matrix = std::make_shared<ContextualSimilarityMatrix>();

	sim_matrix->m_similarity.resize({
		s_vectors->get().attr(PY_SIZE).cast<ssize_t>(),
		t_vectors->get().attr(PY_SIZE).cast<ssize_t>()});

	const py::object tfm_vectors_t = s_vectors->get().attr(PY_TRANSFORM)(t_vectors->get());
	m_metric.vector_metric(s_vectors->get(), tfm_vectors_t, sim_matrix->m_similarity);

	if (m_matcher_factory->needs_magnitudes()) {
		sim_matrix->m_magnitudes_s = xt::pyarray<float>(
			s_vectors->get().attr(PY_MAGNITUDES).cast<py::array_t<float>>());
		sim_matrix->m_magnitudes_t = xt::pyarray<float>(
			tfm_vectors_t.attr(PY_MAGNITUDES).cast<py::array_t<float>>());
	}

	sim_matrix->clip();

	if (m_query->debug_hook().has_value()) {
		sim_matrix->call_hook(m_query);
	}

	return sim_matrix;
}

SimilarityMatrixRef ContextualEmbeddingSimilarityMatrixFactory::create(
	const EmbeddingType p_embedding_type,
	const DocumentRef &p_document) {

	if (p_embedding_type != CONTEXTUAL) {
		throw std::runtime_error("wrong embedding type for contextual embedding similarity matrix");
	}

	py::gil_scoped_acquire acquire;
	return create_with_py_context(p_document);
}

void ContextualSimilarityMatrix::call_hook(
	const QueryRef &p_query) const {

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
	data["columns"] = gen_columns;

	(*p_query->debug_hook())("contextual_similarity_matrix", data);
}