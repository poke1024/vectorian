#include "result_set.h"

py::list ResultSet::best_n(ssize_t p_count) const {

	std::vector sorted(m_matches);
	std::sort_heap(sorted.begin(), sorted.end(), Match::is_worse());

	if (p_count < 0) {
		return sorted;
	}

	py::list matches;

	if (!sorted.empty()) {
		for (auto i = sorted.begin(); i != sorted.begin() + std::min(m_matches.size(), p_count); i++) {
			matches.append(*i);
		}
	}

	return matches;
}
