#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__

#include "metric/metric.h"

class ContextualEmbedding;
typedef std::shared_ptr<ContextualEmbedding> ContextualEmbeddingRef;

class ContextualEmbeddingMetric : public Metric {
protected:
	const ContextualEmbeddingRef m_embedding;
	const py::dict m_options;
	const py::dict m_alignment_def;

public:
	ContextualEmbeddingMetric(
		const EmbeddingRef &p_embedding,
		const py::dict &p_sent_metric_def) :

		m_embedding(std::dynamic_pointer_cast<ContextualEmbedding>(p_embedding)),
		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()) {
	}

	inline const py::dict &options() const {
		return m_options;
	}

	virtual MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

	virtual const std::string &name() const;
};

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_METRIC_H__
