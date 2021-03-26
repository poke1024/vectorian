#include "document.h"
#include "match/match.h"
#include "match/match_impl.h"
#include "match/region.h"

template<template<typename> typename C>
struct MatchDigest::compare {
	inline bool operator()(
		const MatchDigest &a,
		const MatchDigest &b) const {

		if (a.document == b.document) {
			if (C<int32_t>()(a.slice_id, b.slice_id)) {
				return true;
			} else {
				return a.flow.get() < b.flow.get();

				/*return std::lexicographical_compare(
					a.match.begin(), a.match.end(),
					b.match.begin(), b.match.end());*/
			}
		} else {
			PPK_ASSERT(a.document.get() && b.document.get());
			if (C<int64_t>()(a.document->id(), b.document->id())) {
				return true;
			}
		}

		return false;
	}
};

Match::Match(
	const MatcherRef &p_matcher,
	MatchDigest &&p_digest,
	const float p_score) :

	m_matcher(p_matcher),
	m_digest(p_digest),
	m_score(p_score) {
}

Match::Match(
	const MatcherRef &p_matcher,
	const DocumentRef &p_document,
	const int32_t p_slice_id,
	const FlowRef<int16_t> &p_flow,
	const float p_score) :

	m_matcher(p_matcher),
	m_digest(MatchDigest(p_document, p_slice_id, p_flow)),
	m_score(p_score) {
}

py::dict Match::flow_to_py() const {
	return m_digest.flow->to_py();
}

Slice Match::slice() const {
	const auto &level = query()->slice_strategy().level;
	return document()->spans(level)->slice(slice_id());
}

py::list Match::regions(const int window_size) const {
	PPK_ASSERT(document().get() != nullptr);
	return flow()->py_regions(this, window_size);
}

py::list Match::omitted() const {
	return flow()->py_omitted(this);
}

