#ifndef __VECTORIAN_METRIC_H__
#define __VECTORIAN_METRIC_H__

#include "common.h"

class Metric : public std::enable_shared_from_this<Metric> {
	// a point could be made that this is only a partial
	// metric, since it's only between corpus and needle
	// (and not between corpus and corpus), but let's
	// keep names simple.

public:
	virtual ~Metric() {
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document) = 0;

	virtual const std::string &name() const = 0;

	virtual const std::string &origin(int p_token_id_s, int p_query_token_index) const {
		return name();
	}
};

typedef std::shared_ptr<Metric> MetricRef;

class ExternalMetric : public Metric {
	const std::string m_name;

public:
	ExternalMetric(const std::string &p_name) : m_name(p_name) {
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document) {

		throw std::runtime_error(
			"ExternalMetric cannot create matchers");
	}

	virtual const std::string &name() const {
		return m_name;
	}
};

typedef std::shared_ptr<ExternalMetric> ExternalMetricRef;

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
