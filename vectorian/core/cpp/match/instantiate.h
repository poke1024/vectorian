#include "query.h"

MatcherFactoryRef create_matcher_factory(
	const QueryRef &p_query,
	const py::dict &sent_metric_def);
