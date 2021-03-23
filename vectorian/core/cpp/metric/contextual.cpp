#include "result_set.h"
#include "metric/contextual.h"
#include "metric/alignment.h"
#include "embedding/contextual.h"
#include "query.h"
#include "document.h"
#include "metric/factory.h"
#include "slice/contextual.h"

SimilarityMatrixRef ContextualEmbeddingSimilarityMatrixFactory::create(
	const DocumentRef &p_document) {

	py::gil_scoped_acquire acquire;

	const auto embedding_name = m_query->vocabulary()->embedding_manager()->get_py_name(m_embedding_index);

	const auto &cache = m_query->vectors_cache();

	const HandleRef t_vectors = cache.open(
		p_document->get_contextual_embedding_vectors(embedding_name));
	const HandleRef s_vectors = cache.open(
		m_query->get_contextual_embedding_vectors(embedding_name));

	// compute a n x m matrix, (n: number of tokens in document, m: number of tokens in needle)
	// might offload this to GPU. use this as basis for ContextualEmbeddingSlice.

	const auto sim_matrix = std::make_shared<SimilarityMatrix>();

	sim_matrix->m_similarity.resize({
		s_vectors->get().attr("size").cast<ssize_t>(),
		t_vectors->get().attr("size").cast<ssize_t>()});

	m_metric.vector_metric(*s_vectors, *t_vectors, sim_matrix->m_similarity);

	return sim_matrix;
}
