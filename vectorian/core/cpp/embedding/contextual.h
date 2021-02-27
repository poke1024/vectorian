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

	virtual void update_map(
		std::vector<token_t> &p_map,
		const std::vector<std::string> &p_tokens,
		const size_t p_offset) const {

		// nothing.
	}

	virtual MetricRef create_metric(
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<MappedTokenIdArray> &p_vocabulary_to_embedding, // ignored
		const std::vector<Token> &p_needle) {

		const auto m = std::make_shared<ContextualEmbeddingMetric>(
			shared_from_this(),
			p_sent_metric_def);

		return m;
	}

	inline const py::object &compute_embedding_callback() const {
		return m_compute_embedding_callback;
	}
};

typedef std::shared_ptr<ContextualEmbedding> ContextualEmbeddingRef;

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_H__
