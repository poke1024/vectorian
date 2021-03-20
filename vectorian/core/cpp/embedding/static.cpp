#include "embedding/static.h"
#include "metric/static.h"
#include "query.h"

Needle::Needle(
	const QueryRef &p_query) :

	m_needle(p_query->tokens()) {

	const auto &needle = *m_needle;

	m_token_ids.resize({needle.size()});
	for (size_t i = 0; i < needle.size(); i++) {
		m_token_ids(i) = needle[i].id;
	}
}

StaticEmbedding::StaticEmbedding(
	py::object p_embedding_factory,
	py::list p_tokens) :

	Embedding(p_embedding_factory.attr("name").cast<std::string>()),
	m_size(0) {

	m_vectors = p_embedding_factory.attr("get_embeddings")(p_tokens);
	m_size = m_vectors.attr("size").cast<size_t>();
}

MetricRef StaticEmbedding::create_metric(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const py::dict &p_sent_metric_def,
	const std::vector<EmbeddingRef> &p_embeddings) {

	std::vector<StaticEmbeddingRef> embeddings;
	for (auto e : p_embeddings) {
		embeddings.push_back(std::dynamic_pointer_cast<StaticEmbedding>(e));
	}

	const auto metric = std::make_shared<StaticEmbeddingMetricAtom>(
		embeddings,
		p_sent_metric_def);

	metric->initialize(
		p_query,
		p_metric);

	return metric;
}
