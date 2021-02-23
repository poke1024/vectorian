#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"

class FastMetric : public Metric {
protected:
	MatrixXf m_similarity;
	const EmbeddingRef m_embedding;
	const MetricModifiers m_modifiers;
	bool m_similarity_depends_on_pos;

public:
	FastMetric(
		const EmbeddingRef &p_embedding,
		const MetricModifiers &p_modifiers) :

		m_embedding(p_embedding),
		m_modifiers(p_modifiers),
		m_similarity_depends_on_pos(false) {

		if (p_modifiers.pos_mismatch_penalty != 0.0f) {
			m_similarity_depends_on_pos = true;
		} else {
			for (auto x : p_modifiers.t_pos_weights) {
				if (x != 1.0f) {
					m_similarity_depends_on_pos = true;
					break;
				}
			}
		}
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

	inline float pos_weight_for_t(int i) const {
		return m_modifiers.t_pos_weights[i];
	}

	inline bool similarity_depends_on_pos() const {
		return m_similarity_depends_on_pos;
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const;
};

typedef std::shared_ptr<FastMetric> FastMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
