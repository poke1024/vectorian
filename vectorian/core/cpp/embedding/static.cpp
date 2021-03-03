#include "embedding/static.h"
#include "metric/static.h"

MetricRef StaticEmbedding::create_metric(
	const WordMetricDef &p_metric,
	const py::dict &p_sent_metric_def,
	const VocabularyToEmbedding &p_vocabulary_to_embedding,
	const std::vector<Token> &p_needle) {

	return std::make_shared<StaticEmbeddingMetric>(
		std::dynamic_pointer_cast<StaticEmbedding>(shared_from_this()),
		p_metric,
		p_sent_metric_def,
		p_vocabulary_to_embedding,
		p_needle);
}
