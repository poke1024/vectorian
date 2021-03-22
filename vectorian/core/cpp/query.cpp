#include "common.h"
#include "query.h"
#include "match/matcher.h"
#include "result_set.h"

ResultSetRef Query::match(
	const DocumentRef &p_document) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), min_score());

	for (const auto &metric : m_metrics) {
		const auto matcher = metric->matcher_factory()->create_matcher(
			shared_from_this(), metric, p_document);

		matcher->initialize();

		{
			py::gil_scoped_release release;
			matcher->match(matches);
		}
	}

	return matches;
}
