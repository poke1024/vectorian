#include "result_set.h"

py::list ResultSet::best_n(size_t p_count) const {

	std::vector sorted(m_matches);
	std::sort_heap(sorted.begin(), sorted.end(), Match::is_worse());

	py::list matches;

	if (!sorted.empty()) {

		for (auto i = sorted.begin(); i != sorted.begin() + std::min(m_matches.size(), p_count); i++) {
			matches.append(*i);
		}
	}

	return matches;
}
