#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__

#include "metric/metric.h"
#include "embedding/contextual.h"

class ContextualSimilarityMatrix : public SimilarityMatrix {
public:
	virtual void call_hook(
		const QueryRef &p_query) const final;

	virtual SimilarityMatrixRef clone_empty() const final {
		return std::make_shared<ContextualSimilarityMatrix>();
	}
};

typedef std::shared_ptr<ContextualSimilarityMatrix> ContextualSimilarityMatrixRef;


class ContextualEmbeddingSimilarityMatrixFactory : public SimilarityMatrixFactory {
	const QueryRef m_query;
	const WordMetricDef m_metric;
	const MatcherFactoryRef m_matcher_factory;
	const size_t m_embedding_index;

	const py::str PY_SIZE;
	const py::str PY_MAGNITUDES;

	SimilarityMatrixRef build_similarity_matrix(
		const std::vector<StaticEmbeddingRef> &p_embeddings);

	void compute_magnitudes(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const SimilarityMatrixRef &p_matrix);

	SimilarityMatrixRef create_with_py_context(
		const DocumentRef &p_document);

public:
	ContextualEmbeddingSimilarityMatrixFactory(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const MatcherFactoryRef &p_matcher_factory,
		const size_t p_embedding_index);

	virtual SimilarityMatrixRef create(
		const EmbeddingType p_embedding_type,
		const DocumentRef &p_document);
};

typedef std::shared_ptr<ContextualEmbeddingSimilarityMatrixFactory> ContextualEmbeddingSimilarityMatrixFactoryRef;


class ContextualEmbeddingMetric : public Metric {
	const SimilarityMatrixFactoryRef m_factory;

public:
	inline ContextualEmbeddingMetric(
		const std::string &p_name,
		const SimilarityMatrixFactoryRef &p_matrix_factory,
		const MatcherFactoryRef &p_matcher_factory) :

		Metric(p_name, p_matcher_factory, false),
		m_factory(p_matrix_factory) {
	}

	const SimilarityMatrixFactoryRef &matrix_factory() const {
		return m_factory;
	}
};

typedef std::shared_ptr<ContextualEmbeddingMetric> ContextualEmbeddingMetricRef;

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
