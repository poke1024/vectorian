#ifndef __VECTORIAN_MATCH_COMPARE_H__
#define __VECTORIAN_MATCH_COMPARE_H__

#include "query.h"
#include "document.h"

inline int Match::_pos_filter() const {
    return query()->ignore_determiners() ?
        document()->vocabulary()->det_pos() : -1;
}

template<typename Scores>
void Match::compute_scores(const Scores &p_scores, int p_len_s) {
    const auto &match = m_digest.match;

    if (m_scores.empty() && !match.empty()) {
        const auto token_at = sentence().token_at;

        int end = 0;
        for (auto m : match) {
            end = std::max(end, int(m));
        }

        const auto sentence_scores = p_scores.create_sentence_scores(
            token_at, p_len_s, _pos_filter());
        m_scores.reserve(match.size());

        int i = 0;
        for (auto m : match) {
            if (m >= 0) {
                m_scores.push_back(TokenScore{
                    sentence_scores.similarity(m, i),
                    sentence_scores.weight(m, i)});
            } else {
                m_scores.push_back(TokenScore{0.0f, 0.0f});
            }
            i++;
        }
    }
}

template<template<typename> typename C>
struct Match::compare_by_score {
	inline bool operator()(
		const MatchRef &a,
		const MatchRef &b) const {

		if (C<float>()(a->score(), b->score())) {
			return true;
		} else if (a->score() == b->score()) {

			if (a->document() == b->document()) {

				if (C<int32_t>()(a->sentence_id(), b->sentence_id())) {
					return true;
				} else {

					return std::lexicographical_compare(
						a->match().begin(), a->match().end(),
						b->match().begin(), b->match().end());

				}
			} else {

				PPK_ASSERT(a->document().get() && b->document().get());

				if (C<int64_t>()(a->document()->id(), b->document()->id())) {
					return true;
				}
			}
		}
		return false;
	}
};

#endif // __VECTORIAN_MATCH_COMPARE_H__
