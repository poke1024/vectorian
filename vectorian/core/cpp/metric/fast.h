#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"

class FastMetric : public Metric {
protected:
	MatrixXf m_similarity;
	const EmbeddingRef m_embedding;
	const MetricModifiers m_modifiers;

public:
	FastMetric(
		const EmbeddingRef &p_embedding,
		const MetricModifiers &p_modifiers) :

		m_embedding(p_embedding),
		m_modifiers(p_modifiers) {
	}

	inline MatrixXf &w_similarity() {
		return m_similarity;
	}

	inline const MatrixXf &similarity() const {
		return m_similarity;
	}

	inline const MetricModifiers &modifiers() const {
		return m_modifiers;
	}

	inline float pos_weight(int tag) const {
		const auto &posw = m_modifiers.pos_weights;
		const auto w = posw.find(tag);
		if (w != posw.end()) {
			return w->second;
		} else {
			return 1.0f;
		}
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const;
};

typedef std::shared_ptr<FastMetric> FastMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
