#ifndef __VECTORIAN_COMPOSITE_METRIC_H__
#define __VECTORIAN_COMPOSITE_METRIC_H__

#include "metric/fast.h"

class CompositeMetric : public FastMetric {
	const FastMetricRef m_a;
	const FastMetricRef m_b;
	std::string m_name;
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_is_from_a;

public:
	CompositeMetric(
		const MetricRef &p_a,
		const MetricRef &p_b,
		float t) :

		FastMetric(
			EmbeddingRef(),
			std::dynamic_pointer_cast<FastMetric>(p_a)->modifiers()),
		m_a(std::dynamic_pointer_cast<FastMetric>(p_a)),
		m_b(std::dynamic_pointer_cast<FastMetric>(p_b)) {

		const float t2 = 2.0f * t; // [0, 2] for linear interpolation
		const float k = 1.0f + std::abs(t2 - 1.0f); // [1, 2] for normalization

		const MatrixXf arg_a = m_a->similarity() * ((2.0f - t2) / k);
		this->m_similarity = arg_a.cwiseMax(m_b->similarity() * (t2 / k));

		m_is_from_a = this->m_similarity.cwiseEqual(arg_a);

		if (t == 0.0f) {
			m_name = m_a->name();
		} else if (t == 1.0f) {
			m_name = m_b->name();
		} else {
			char buf[32];
			snprintf(buf, 32, "%.2f", t);

			std::ostringstream s;
			s << m_a->name() << " + " << m_b->name() << " @" << buf;
			m_name = s.str();
		}
	}

	virtual const std::string &name() const {
		return m_name;
	}

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return m_is_from_a(p_token_id_s, p_query_token_index) ?  m_a->name() :  m_b->name();
	}
};

typedef std::shared_ptr<CompositeMetric> CompositeMetricRef;

#endif // __VECTORIAN_COMPOSITE_METRIC_H__
