#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_H__

#include "common.h"
#include "embedding/embedding.h"

class ContextualEmbedding : public TokenEmbedding {
public:
	ContextualEmbedding(
		const std::string &p_name) :

		TokenEmbedding(p_name) {
	}

	/*virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_word_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<EmbeddingRef>&) {

		const auto metric = std::make_shared<ContextualEmbeddingMetric>(
			shared_from_this(),
			p_sent_metric_def);

		metric->initialize(p_query, p_word_metric);

		return metric;
	}*/
};

typedef std::shared_ptr<ContextualEmbedding> ContextualEmbeddingRef;

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_H__
