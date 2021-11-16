#ifndef __VECTORIAN_METRIC_FACTORY__
#define __VECTORIAN_METRIC_FACTORY__

#include "match/matcher_impl.h"

template<typename SliceFactory, typename Aligner, typename Finalizer>
MatcherRef make_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const BoosterRef &p_booster,
	const MetricRef &p_metric,
	const SliceFactory &p_factory,
	Aligner &&p_aligner,
	const Finalizer &p_finalizer) {

	return std::make_shared<MatcherImpl<SliceFactory, Aligner, Finalizer>>(
		p_query, p_document, p_booster, p_metric, std::move(p_aligner), p_finalizer, p_factory);
}

#endif // __VECTORIAN_METRIC_FACTORY__
