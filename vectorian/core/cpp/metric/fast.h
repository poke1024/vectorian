#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"

class FastMetric : public Metric {
protected:
	MatrixXf m_similarity;
	const EmbeddingRef m_embedding;
	const float m_pos_mismatch_penalty;
	const float m_similarity_falloff;
	const float m_similarity_threshold;
	const POSWMap m_pos_weights;

public:
	FastMetric(
		const EmbeddingRef &p_embedding,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights) :

		m_embedding(p_embedding),
		m_pos_mismatch_penalty(p_pos_mismatch_penalty),
		m_similarity_falloff(p_similarity_falloff),
		m_similarity_threshold(p_similarity_threshold),
		m_pos_weights(p_pos_weights) {
	}

	inline MatrixXf &w_similarity() {
		return m_similarity;
	}

	inline const MatrixXf &similarity() const {
		return m_similarity;
	}

	inline float pos_mismatch_penalty() const {
		return m_pos_mismatch_penalty;
	}

	inline float similarity_threshold() const {
		return m_similarity_threshold;
	}

	inline float similarity_falloff() const {
		return m_similarity_falloff;
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	inline float pos_weight(int tag) const {
		const auto w = m_pos_weights.find(tag);
		if (w != m_pos_weights.end()) {
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
