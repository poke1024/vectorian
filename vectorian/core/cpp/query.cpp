#include "common.h"
#include "query.h"
#include "match/matcher.h"
#include "result_set.h"

ResultSetRef Query::match(
	const DocumentRef &p_document) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), min_score());

	for (const auto &strategy : m_match_strategies) {
		const auto matcher = strategy.matcher_factory->create_matcher(p_document);

		matcher->initialize();

		{
			py::gil_scoped_release release;
			matcher->match(matches);
		}
	}

	return matches;
}
