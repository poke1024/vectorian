#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include <xtensor/xadapt.hpp>

class StaticSimilarityMatrix : public SimilarityMatrix {
public:
	virtual void call_hook(
		const QueryRef &p_query) const final;

	virtual SimilarityMatrixRef clone_empty() const final {
		return std::make_shared<StaticSimilarityMatrix>();
	}
};

class StaticEmbeddingMetric : public Metric {
	const SimilarityMatrixRef m_matrix;

public:
	inline StaticEmbeddingMetric(
		const std::string &p_name,
		const SimilarityMatrixRef &p_matrix,
		const MatcherFactoryRef &p_matcher_factory) :

		Metric(p_name, p_matcher_factory, true),
		m_matrix(p_matrix) {
	}

	virtual bool is_based_on_static_embedding() const {
		return true;
	}

	inline const SimilarityMatrixRef &matrix() const {
		return m_matrix;
	}
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;


class StaticEmbeddingMatcherFactoryFactory {
	py::dict m_sent_metric_def;

	inline const py::dict &sent_metric_def() const {
		return m_sent_metric_def;
	}

	inline py::dict alignment_def() const {
		return m_sent_metric_def["alignment"].cast<py::dict>();
	}

public:
	StaticEmbeddingMatcherFactoryFactory(
		const py::dict &p_sent_metric_def) :

		m_sent_metric_def(p_sent_metric_def) {
	}

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);
};


class StaticEmbeddingSimilarityMatrixFactory : public SimilarityMatrixFactory {
	const QueryRef m_query;
	const WordMetricDef m_metric;
	const MatcherFactoryRef m_matcher_factory;
	const size_t m_embedding_index;
	SimilarityMatrixRef m_static_matrix;

	SimilarityMatrixRef build_static_similarity_matrix(
		const std::vector<StaticEmbeddingRef> &p_embeddings);

	void compute_magnitudes(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const SimilarityMatrixRef &p_matrix);

public:
	StaticEmbeddingSimilarityMatrixFactory(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const MatcherFactoryRef &p_matcher_factory,
		const size_t p_embedding_index);

	virtual SimilarityMatrixRef create(
		const EmbeddingType p_embedding_type,
		const DocumentRef &p_document);
};

typedef std::shared_ptr<StaticEmbeddingSimilarityMatrixFactory> StaticEmbeddingSimilarityMatrixFactoryRef;

#endif // __VECTORIAN_FAST_METRIC_H__
