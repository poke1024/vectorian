#ifndef __VECTORIAN_MATCH_COMPARE_H__
#define __VECTORIAN_MATCH_COMPARE_H__

#include "query.h"
#include "document.h"
#include "slice/static.h"

template<typename SliceFactory>
void Match::compute_scores(
	const SliceFactory &p_factory,
	const int p_len_s, const int p_len_t) {

    const auto &match = m_digest.match;

    if (m_scores.empty() && !match.empty()) {
        const auto token_at = sentence().idx;

        int end = 0;
        for (auto m : match) {
            end = std::max(end, int(m));
        }

		const Token *s_tokens = document()->tokens()->data();
		const Token *t_tokens = query()->tokens()->data();

        const auto slice = p_factory.create_slice(
            TokenSpan{s_tokens + token_at, p_len_s},
            TokenSpan{t_tokens, p_len_t});
        m_scores.reserve(match.size());

        int i = 0;
        for (auto m : match) {
            if (m >= 0) {
                m_scores.emplace_back(TokenScore{
                    slice.unmodified_similarity(m, i),
                    1.0f}); // FIXME; was: slice.weight(m, i)
            } else {
                m_scores.emplace_back(
                    TokenScore{0.0f, 0.0f});
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
