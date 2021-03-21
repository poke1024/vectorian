#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_H__

#include "common.h"
#include "embedding/embedding.h"
#include "metric/contextual.h"

class ContextualEmbedding : public Embedding {
	const py::object m_compute_embedding_callback;

public:
	ContextualEmbedding(
		const std::string &p_name,
		py::object p_compute) :

		Embedding(p_name),
		m_compute_embedding_callback(p_compute) {
	}

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_word_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<EmbeddingRef>&) {

		const auto metric = std::make_shared<ContextualEmbeddingMetric>(
			shared_from_this(),
			p_sent_metric_def);

		metric->initialize(p_query, p_word_metric);

		return metric;
	}

	inline const py::object &compute_embedding_callback() const {
		return m_compute_embedding_callback;
	}
};

typedef std::shared_ptr<ContextualEmbedding> ContextualEmbeddingRef;

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_H__
