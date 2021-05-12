#ifndef __VECTORIAN_MATCH_COMPARE_H__
#define __VECTORIAN_MATCH_COMPARE_H__

#include "query.h"
#include "document.h"
#include "slice/static.h"

template<template<typename> typename C>
struct Match::compare_by_score {
	inline bool operator()(
		const MatchRef &a,
		const MatchRef &b) const {

		if (C<float>()(a->score().normalized(), b->score().normalized())) {
			return true;
		} else if (a->score() == b->score()) {

			if (a->document() == b->document()) {

				if (C<int32_t>()(a->slice_id(), b->slice_id())) {
					return true;
				} else {

					return a->flow().get() < b->flow().get();

					/*return std::lexicographical_compare(
						a->match().begin(), a->match().end(),
						b->match().begin(), b->match().end());*/

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
