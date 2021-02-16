#include "result_set.h"

py::list ResultSet::best_n(ssize_t p_count) const {

	std::vector sorted(m_matches);
	py::list matches;

	if (!sorted.empty()) {
		const size_t n = p_count < 0 ?
			m_matches.size() :
			std::min(m_matches.size(), static_cast<size_t>(p_count));
		std::partial_sort(sorted.begin(), sorted.begin() + n, sorted.end());
		for (auto i = sorted.begin(); i != sorted.begin() + n; i++) {
			matches.append(*i);
		}
	}

	return matches;
}
