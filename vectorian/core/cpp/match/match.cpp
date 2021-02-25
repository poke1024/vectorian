#include "utils.h"
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
			if (C<int32_t>()(a.sentence_id, b.sentence_id)) {
				return true;
			} else {

				return std::lexicographical_compare(
					a.match.begin(), a.match.end(),
					b.match.begin(), b.match.end());

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
	const int32_t p_sentence_id,
	const std::vector<int16_t> &p_match,
	const float p_score) :

	m_matcher(p_matcher),
	m_digest(MatchDigest(p_document, p_sentence_id, p_match)),
	m_score(p_score) {
}

py::dict Match::py_assignment() const {
	py::dict d;

	// "idx" is index into aligned doc sentence token for each query token.
	{
		const std::vector<ssize_t> shape = {
			static_cast<ssize_t>(m_digest.match.size())};
		d["idx"] = py::array_t<int16_t>(
	        shape,                  // shape
	        {sizeof(int16_t)},      // strides
	        m_digest.match.data()); // data pointer
	}

	{
		const std::vector<ssize_t> shape = {
			static_cast<ssize_t>(m_scores.size())};
		const uint8_t* const data =
			reinterpret_cast<const uint8_t*>(m_scores.data());

		d["sim"] = PY_ARRAY_MEMBER(TokenScore, similarity);
		d["w"] = PY_ARRAY_MEMBER(TokenScore, weight);
	}

	return d;
}


const Sentence &Match::sentence() const {
	return  document()->sentence(sentence_id());
}

py::list Match::regions() const {
	PPK_ASSERT(document().get() != nullptr);

	const std::string &s_text = document()->text();
	const std::string &t_text = query()->text();
	const auto &s_tokens_ref = document()->tokens();
	const auto &t_tokens_ref = query()->tokens();
	const std::vector<Token> &s_tokens = *s_tokens_ref.get();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	const auto token_at = sentence().token_at;

	const auto &match = this->match();
	const auto &scores = m_scores;

	PPK_ASSERT(match.size() > 0);
	PPK_ASSERT(match.size() == scores.size());

	int match_0 = 0;
	for (auto m : match) {
		if (m >= 0) {
			match_0 = m;
			break;
		}
	}

	constexpr int window_size = 10;
	int32_t last_anchor = std::max(0, token_at + match_0 - window_size);
	bool last_matched = false;

	py::list regions;
	const int32_t n = static_cast<int32_t>(match.size());

	const TokenFilter &token_filter = query()->token_filter();
	std::vector<int16_t> index_map;
	if (!token_filter.all()) {
		int16_t k = 0;
		index_map.resize(sentence().n_tokens);
		for (int32_t i = 0; i < sentence().n_tokens; i++) {
			index_map[k] = i;
			if (token_filter(s_tokens.at(token_at + i))) {
				k++;
			}
		}
	}

	struct MatchLoc {
		int32_t i;
		int32_t match_at_i;
	};
	std::vector<MatchLoc> locs;
	locs.reserve(n);

	for (int32_t i = 0; i < n; i++) {
	    int match_at_i = match[i];

		if (match_at_i < 0) {
			continue;
		}

		if (!index_map.empty()) {
			match_at_i = index_map[match_at_i];
		}

		locs.emplace_back(MatchLoc{i, match_at_i});
	}

	std::sort(locs.begin(), locs.end(), [] (const MatchLoc &a, const MatchLoc &b) {
		return a.match_at_i < b.match_at_i;
	});

	for (const auto &loc : locs) {
		const auto i = loc.i;
		const auto match_at_i = loc.match_at_i;

		const auto &s = s_tokens.at(token_at + match_at_i);
		const auto &t = t_tokens.at(i);

		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		if (s.idx > idx0) {

			// this is for displaying the relative (!) penalty in the UI.

			float p;

			if (last_matched) {
				p = m_matcher->gap_cost(token_at + match_at_i - last_anchor);
			} else {
				p = 0.0f;
			}

			regions.append(std::make_shared<Region>(
				s_text.substr(idx0, s.idx - idx0), p));
		}

		regions.append(std::make_shared<MatchedRegion>(
			scores[i],
			s_text.substr(s.idx, s.len),
			t_text.substr(t.idx, t.len),
			query()->vocabulary(),
			TokenRef{s_tokens_ref, token_at + match_at_i},
			TokenRef{t_tokens_ref, i},
			metric()->origin(s.id, i)
		));

		last_anchor = token_at + match_at_i + 1;
		last_matched = true;
	}

	const int32_t up_to = std::min(last_anchor + window_size, int32_t(s_tokens.size() - 1));
	if (up_to > last_anchor) {
		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		regions.append(std::make_shared<Region>(
			s_text.substr(idx0, s_tokens.at(up_to).idx - idx0)));
	}

	return regions;
}

py::list Match::omitted() const {

	const auto &t_tokens_ref = query()->tokens();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();
	const std::string &t_text = query()->text();

	py::list not_used;

	const auto &match = this->match();
	for (int i = 0; i < int(match.size()); i++) {
		if (match[i] < 0) {
			const auto &t = t_tokens.at(i);
			not_used.append(py::str(t_text.substr(t.idx, t.len)));
		}
	}

	return not_used;
}
