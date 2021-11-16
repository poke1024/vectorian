#include "match/matcher.h"
#include "metric/metric.h"

Matcher::~Matcher() {
}

MatcherRef MatcherFactory::create_matcher(
	const QueryRef &p_query,
	const MetricRef &p_metric,
	const DocumentRef &p_document,
	const BoosterRef &p_booster) const {

	if (p_metric->is_based_on_static_embedding()) {
		PPK_ASSERT(m_static_factory.get());
		return m_static_factory->create_matcher(
			p_query, p_metric, p_document, p_booster, this->m_options);
	} else {
		PPK_ASSERT(m_contextual_factory.get());
		return m_contextual_factory->create_matcher(
			p_query, p_metric, p_document, p_booster, this->m_options);
	}
}
