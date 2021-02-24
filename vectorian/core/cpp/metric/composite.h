#ifndef __VECTORIAN_COMPOSITE_METRIC_H__
#define __VECTORIAN_COMPOSITE_METRIC_H__

#include "metric/static.h"

class MinMaxMetric : public StaticEmbeddingMetric {
	const StaticEmbeddingMetricRef m_a;
	const StaticEmbeddingMetricRef m_b;
	std::string m_name;
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_is_from_a;

public:
	template<typename F>
	MinMaxMetric(
		const MetricRef &p_a,
		const MetricRef &p_b,
		const F &p_f,
		const char *p_name) :

		StaticEmbeddingMetric(
			EmbeddingRef(),
			std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_a)->options()),
		m_a(std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_a)),
		m_b(std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_b)) {

		this->m_similarity = m_a->similarity().cwiseMax(m_b->similarity());
		m_is_from_a = this->m_similarity.cwiseEqual(m_a->similarity());

		std::ostringstream s;
		s << p_name << "(" << m_a->name() << ", " << m_b->name() << ")";
		m_name = s.str();
	}

	virtual const std::string &name() const {
		return m_name;
	}

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return m_is_from_a(p_token_id_s, p_query_token_index) ?  m_a->name() :  m_b->name();
	}
};


class MaxMetric : public MinMaxMetric {
public:
	MaxMetric(
		const MetricRef &p_a,
		const MetricRef &p_b) : MinMaxMetric(p_a, p_b, [] (const auto &a, const auto &b) {
			return a.cwiseMax(b);
		}, "max") {
	}
};

typedef std::shared_ptr<MaxMetric> MaxMetricRef;


class MinMetric : public MinMaxMetric {
public:
	MinMetric(
		const MetricRef &p_a,
		const MetricRef &p_b) : MinMaxMetric(p_a, p_b, [] (const auto &a, const auto &b) {
			return a.cwiseMin(b);
		}, "min") {
	}
};

typedef std::shared_ptr<MinMetric> MinMetricRef;


class LerpMetric : public StaticEmbeddingMetric {
	const StaticEmbeddingMetricRef m_a;
	const StaticEmbeddingMetricRef m_b;
	std::string m_name;

public:
	LerpMetric(
		const MetricRef &p_a,
		const MetricRef &p_b,
		float t) :

		StaticEmbeddingMetric(
			EmbeddingRef(),
			std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_a)->options()),
		m_a(std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_a)),
		m_b(std::dynamic_pointer_cast<StaticEmbeddingMetric>(p_b)) {

		this->m_similarity = (m_a->similarity() * (1 - t)) + (m_b->similarity() * t);

		std::ostringstream s;
		s << "lerp(" << m_a->name() << ", " << m_b->name() << ", " << t << ")";
		m_name = s.str();
	}

	virtual const std::string &name() const {
		return m_name;
	}

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return m_name;
	}
};

typedef std::shared_ptr<LerpMetric> LerpMetricRef;

#endif // __VECTORIAN_COMPOSITE_METRIC_H__
