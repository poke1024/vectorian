#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__

#include "metric/metric.h"

class ContextualEmbedding;
typedef std::shared_ptr<ContextualEmbedding> ContextualEmbeddingRef;

class ContextualEmbeddingMetric : public Metric {
protected:
	const ContextualEmbeddingRef m_embedding;
	const py::dict m_sent_metric_def;
	MatcherFactoryRef m_matcher_factory;

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query,
		const WordMetricDef &p_word_metric);

	inline py::dict alignment_def() const {
		return m_sent_metric_def["alignment"].cast<py::dict>();
	}

public:
	ContextualEmbeddingMetric(
		const EmbeddingRef &p_embedding,
		const py::dict &p_sent_metric_def) :

		m_embedding(std::dynamic_pointer_cast<ContextualEmbedding>(p_embedding)),
		m_sent_metric_def(p_sent_metric_def) {
	}

	void initialize(
		const QueryRef &p_query,
		const WordMetricDef &p_word_metric) {

		m_matcher_factory = create_matcher_factory(p_query, p_word_metric);
	}

	virtual MatcherFactoryRef matcher_factory() const {
		return m_matcher_factory;
	}

	virtual const std::string &name() const;
};

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
