#ifndef __VECTORIAN_RESULT_SET_H__
#define __VECTORIAN_RESULT_SET_H__

#include "common.h"
#include "match/match.h"
#include "match/match_impl.h"

class GroundTruth {
	struct Item {
		int32_t document_id;
		int32_t sentence_id;
	};

	std::vector<Item> m_items;
};

class ResultSet {
public:
	ResultSet(
		const size_t p_max_matches,
		const float p_min_score) :

		m_max_matches(p_max_matches),
		m_min_score(p_min_score) {

		m_matches.reserve(p_max_matches);
	}

	inline float worst_score() const {
		if (m_matches.empty()) {
			return m_min_score;
		} else {
			return m_matches[0]->score();
		}
	}

	inline void add(const MatchRef &p_match) {

		if (p_match->score() > worst_score()) {

			if (m_matches.size() >= m_max_matches) {

				std::pop_heap(
					m_matches.begin(),
					m_matches.end(),
					Match::is_greater());

				m_matches.pop_back();
			}

			m_matches.push_back(p_match);

			std::push_heap(
				m_matches.begin(),
				m_matches.end(),
				Match::is_greater());
		}
	}

	inline size_t size() const {
		return m_matches.size();
	}

	void extend(
		const ResultSet &p_set) {

		m_matches.reserve(
			m_matches.size() + p_set.m_matches.size());

		for (const auto &a : p_set.m_matches) {
			if (m_matches.size() >= m_max_matches) {
				std::pop_heap(
					m_matches.begin(),
					m_matches.end(),
					Match::is_greater());

				m_matches.pop_back();
			}

			m_matches.push_back(a);

			std::push_heap(
				m_matches.begin(),
				m_matches.end(),
				Match::is_greater());
		}
	}

	void extend_and_remove_duplicates(
		const ResultSet &p_set) {

		// not yet implemented, should remove
		// duplicate results on the same sentence.

		 PPK_ASSERT(false);
	}

	py::list best_n(ssize_t p_count) const;

	float precision(const GroundTruth &p_truth) const {
		return 0.0f; // to be implemented
	}

	float recall(const GroundTruth &p_truth) const {
		return 0.0f; // to be implemented
	}

private:
	const QueryRef m_query;

	// a heap such that m_matches[0] contains the
	// match with the worst/lowest score.
	std::vector<MatchRef> m_matches;

	const size_t m_max_matches;
	const float m_min_score;
};

#endif // __VECTORIAN_RESULT_SET_H__
