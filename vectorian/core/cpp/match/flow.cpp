#ifndef __VECTORIAN_FLOW_IMPL_H__
#define __VECTORIAN_FLOW_IMPL_H__

#include "match/match.h"
#include "document.h"

template<typename Index>
py::list OneToOneFlow<Index>::py_regions(const Match *p_match, const int window_size) const {
	const auto &s_tokens_ref = p_match->document()->tokens();
	const auto &t_tokens_ref = p_match->query()->tokens();
	const std::vector<Token> &s_tokens = *s_tokens_ref.get();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	const auto token_at = p_match->slice().idx;

	const auto &match = m_mapping;
	py::list regions;

	if (match.size() < 1) {
		const auto &s0 = s_tokens.at(token_at);
		const auto &s1 = s_tokens.at(std::min(
			static_cast<size_t>(token_at + p_match->slice().len), s_tokens.size() - 1));
		regions.append(std::make_shared<Region>(
			Slice{s0.idx, s1.idx - s0.idx}, 0.0f));
		return regions;
	}

	Index match_0 = 0;
	for (auto m : match) {
		if (m.target >= 0) {
			match_0 = m.target;
			break;
		}
	}

	int32_t last_anchor = std::max(0, token_at + match_0 - window_size);
	bool last_matched = false;

	const int32_t n = static_cast<int32_t>(match.size());

	const TokenFilter &token_filter = p_match->query()->token_filter();
	std::vector<int16_t> index_map;
	if (!token_filter.all()) {
		int16_t k = 0;
		const auto len = p_match->slice().len;
		index_map.resize(len);
		for (int32_t i = 0; i < len; i++) {
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
	    int match_at_i = match[i].target;

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
				p = p_match->matcher()->gap_cost(token_at + match_at_i - last_anchor);
			} else {
				p = 0.0f;
			}

			regions.append(std::make_shared<Region>(
				Slice{idx0, s.idx - idx0}, p));
		}

		regions.append(std::make_shared<MatchedRegion>(
			TokenScore{match[i].similarity, match[i].weight},
			Slice{s.idx, s.len},
			Slice{t.idx, t.len},
			p_match->query()->vocabulary(),
			TokenRef{s_tokens_ref, token_at + match_at_i},
			TokenRef{t_tokens_ref, i},
			p_match->metric()->origin(s.id, i)
		));

		last_anchor = token_at + match_at_i + 1;
		last_matched = true;
	}

	const int32_t up_to = std::min(last_anchor + window_size, int32_t(s_tokens.size() - 1));
	if (up_to > last_anchor) {
		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		regions.append(std::make_shared<Region>(
			Slice{idx0, s_tokens.at(up_to).idx - idx0}));
	}

	return regions;
}

template<typename Index>
py::list OneToOneFlow<Index>::py_omitted(const Match *p_match) const {
	const auto &t_tokens_ref = p_match->query()->tokens();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	py::list not_used;

	const auto &match = m_mapping;
	for (int i = 0; i < int(match.size()); i++) {
		if (match[i].target < 0) {
			const auto &t = t_tokens.at(i);
			not_used.append(Slice{t.idx, t.len}.to_py());
		}
	}

	return not_used;
}

template py::list OneToOneFlow<int16_t>::py_regions(const Match *p_match, const int window_size) const;
template py::list OneToOneFlow<int16_t>::py_omitted(const Match *p_match) const;

#endif // __VECTORIAN_FLOW_IMPL_H__
