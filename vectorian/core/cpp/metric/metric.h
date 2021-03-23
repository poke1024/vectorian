#ifndef __VECTORIAN_METRIC_H__
#define __VECTORIAN_METRIC_H__

#include "common.h"

class SimilarityMatrix {
public:
	xt::pytensor<float, 2> m_similarity;
	xt::pytensor<float, 1> m_magnitudes;
	bool m_needs_magnitudes;

	inline SimilarityMatrix() :

		m_needs_magnitudes(false) {

		m_magnitudes.resize({0});
		PPK_ASSERT(m_magnitudes.shape(0) == 0);
	}

	inline const xt::pytensor<float, 2> &similarity() const {
		return m_similarity;
	}

	inline const xt::pytensor<float, 1> &magnitudes() const {
		return m_magnitudes;
	}

	inline void assert_has_magnitudes() const {
		PPK_ASSERT(m_magnitudes.shape(0) > 0);
	}
};

typedef std::shared_ptr<SimilarityMatrix> SimilarityMatrixRef;


class SimilarityMatrixFactory {
public:
	virtual ~SimilarityMatrixFactory() {
	}

	virtual SimilarityMatrixRef create(const DocumentRef &p_document) = 0;
};

typedef std::shared_ptr<SimilarityMatrixFactory> SimilarityMatrixFactoryRef;


class Metric : public std::enable_shared_from_this<Metric> {
protected:
	const std::string m_name;
	const MatcherFactoryRef m_matcher_factory;
	const bool m_based_on_static_embedding;

public:
	inline Metric(
		const std::string &p_name,
		const MatcherFactoryRef &p_matcher_factory,
		const bool p_based_on_static_embedding) :

		m_name(p_name),
		m_matcher_factory(p_matcher_factory),
		m_based_on_static_embedding(p_based_on_static_embedding) {
	}

	virtual ~Metric() {
	}

	bool is_based_on_static_embedding() const {
		return m_based_on_static_embedding;
	}

	inline const MatcherFactoryRef &matcher_factory() const {
		return m_matcher_factory;
	}

	inline const std::string &name() const {
		return m_name;
	}

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return name();
	}
};

typedef std::shared_ptr<Metric> MetricRef;

/*class ExternalMetric : public Metric {
	const std::string m_name;

public:
	ExternalMetric(const std::string &p_name) : m_name(p_name) {
	}

	virtual MatcherFactoryRef matcher_factory() const {
		throw std::runtime_error(
			"ExternalMetric cannot create matchers");
		return MatcherFactoryRef();
	}

	virtual const std::string &name() const {
		return m_name;
	}
};

typedef std::shared_ptr<ExternalMetric> ExternalMetricRef;*/

inline MetricRef lookup_metric(
	const std::map<std::string, MetricRef> &p_metrics,
	const std::string &p_name) {

	const auto i = p_metrics.find(p_name);
	if (i == p_metrics.end()) {
		throw std::runtime_error(
			std::string("could not find a metric named ") + p_name);
	}
	return i->second;
}

#endif // __VECTORIAN_METRIC_H__
