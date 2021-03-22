#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include <xtensor/xadapt.hpp>

class StaticEmbeddingMetric : public Metric {
public:
	inline StaticEmbeddingMetric(
		const std::string &p_name,
		const SimilarityMatrixRef &p_matrix,
		const MatcherFactoryRef &p_matcher_factory) :

		Metric(p_name, p_matrix, p_matcher_factory) {
	}

	virtual MetricRef clone(const SimilarityMatrixRef &p_matrix) {
		return std::make_shared<StaticEmbeddingMetric>(
			m_name, p_matrix, m_matcher_factory);
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

	SimilarityMatrixRef build_similarity_matrix(
		const std::vector<StaticEmbeddingRef> &p_embeddings);

	void compute_magnitudes(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const SimilarityMatrixRef &p_matrix);

public:
	StaticEmbeddingSimilarityMatrixFactory(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const MatcherFactoryRef &p_matcher_factory,
		const size_t p_embedding_index) :

		m_query(p_query),
		m_metric(p_metric),
		m_matcher_factory(p_matcher_factory),
		m_embedding_index(p_embedding_index) {
	}

	virtual SimilarityMatrixRef create(
		const DocumentRef &p_document);
};

typedef std::shared_ptr<StaticEmbeddingSimilarityMatrixFactory> StaticEmbeddingSimilarityMatrixFactoryRef;

#endif // __VECTORIAN_FAST_METRIC_H__
