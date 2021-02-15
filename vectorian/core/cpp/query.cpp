#include "common.h"
#include "query.h"
#include "matcher.h"
#include "result_set.h"

ResultSetRef Query::match(
	const DocumentRef &p_document) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), min_score());

	const auto me = shared_from_this();

	for (const auto &metric : m_metrics) {
		auto matcher = metric->create_matcher(
			me,
			p_document);
		matcher->match(matches);
	}

	return matches;
}
