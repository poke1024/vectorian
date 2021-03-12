#ifndef __VECTORIAN_FLOW_IMPL_H__
#define __VECTORIAN_FLOW_IMPL_H__

#include "match/match.h"
#include "document.h"
#include <xtensor/xsort.hpp>

template<typename Index>
py::list Flow<Index>::py_regions(
	const Match *p_match,
	const std::vector<Edge> &p_edges,
	const int p_window_size) const {

	const auto &s_tokens_ref = p_match->document()->tokens();
	const auto &t_tokens_ref = p_match->query()->tokens();
	const std::vector<Token> &s_tokens = *s_tokens_ref.get();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	const auto token_at = p_match->slice().idx;

	py::list regions;

	if (p_edges.size() < 1) {
		const auto &s0 = s_tokens.at(token_at);
		const auto &s1 = s_tokens.at(std::min(
			static_cast<size_t>(token_at + p_match->slice().len), s_tokens.size() - 1));
		regions.append(std::make_shared<Region>(
			Slice{s0.idx, s1.idx - s0.idx}, 0.0f));
		return regions;
	}

	std::vector<Edge> edges(p_edges);
	std::sort(edges.begin(), edges.end(), [] (const Edge &a, const Edge &b) {
		if (a.target < b.target) {
			return true;
		} else if (a.target > b.target) {
			return false;
		} else {
			return a.weight.flow > b.weight.flow; // biggest flow first.
		}
	});

	const Index match_0 = edges[0].target;
	auto last_anchor = std::max(0, token_at + match_0 - p_window_size);
	bool last_matched = false;

	//const int32_t n = static_cast<int32_t>(match.size());

	const TokenFilter &token_filter = p_match->query()->token_filter();
	std::vector<Index> index_map;
	if (!token_filter.all()) {
		size_t k = 0;
		const auto len = p_match->slice().len;
		index_map.resize(len);
		for (ssize_t i = 0; i < len; i++) {
			index_map[k] = i;
			if (token_filter(s_tokens.at(token_at + i))) {
				k++;
			}
		}
	}

	/*struct MatchLoc {
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
	});*/

	size_t edge_index = 0;

	while (edge_index < edges.size()) {
		const auto &edge = edges[edge_index];

		auto target = edge.target;
		PPK_ASSERT(target >= 0);
		if (!index_map.empty()) {
			target = index_map[target];
		}

		const auto &s = s_tokens.at(token_at + target);

		const int32_t idx0 = s_tokens.at(last_anchor).idx;
		if (s.idx > idx0) {
			float p;

			if (last_matched) {
				p = p_match->matcher()->gap_cost(
					token_at + target - last_anchor);
			} else {
				p = 0.0f;
			}

			regions.append(std::make_shared<Region>(
				Slice{idx0, s.idx - idx0}, p));
		}

		std::vector<MatchedRegion::HalfEdge> region_edges;

		do {
			const auto source = edges[edge_index].source;
			const auto &t = t_tokens.at(source);
			region_edges.emplace_back(MatchedRegion::HalfEdge(
				edges[edge_index].weight,
				Slice{t.idx, t.len},
				TokenRef{t_tokens_ref, source},
				p_match->metric()->origin(s.id, source)
			));

			edge_index += 1;
		} while (edge_index < edges.size() && edges[edge_index].target == target);

		regions.append(std::make_shared<MatchedRegion>(
			p_match->query()->vocabulary(),
			Slice{s.idx, s.len},
			TokenRef{s_tokens_ref, token_at + target},
			std::move(region_edges)
		));

		last_anchor = token_at + target + 1;
		last_matched = true;
	}

	const auto up_to = std::min(
		last_anchor + p_window_size, int32_t(s_tokens.size() - 1));
	if (up_to > last_anchor) {
		const auto idx0 = s_tokens.at(last_anchor).idx;
		regions.append(std::make_shared<Region>(
			Slice{idx0, s_tokens.at(up_to).idx - idx0}));
	}

	return regions;
}

template<typename Index>
py::list Flow<Index>::py_omitted(
	const Match *p_match,
	const std::vector<HalfEdge> &p_edges) const {

	const auto &t_tokens_ref = p_match->query()->tokens();
	const std::vector<Token> &t_tokens = *t_tokens_ref.get();

	py::list not_used;

	const auto &match = p_edges;
	for (int i = 0; i < int(match.size()); i++) {
		if (match[i].target < 0) {
			const auto &t = t_tokens.at(i);
			not_used.append(Slice{t.idx, t.len}.to_py());
		}
	}

	return not_used;
}

template<typename Index>
py::dict InjectiveFlow<Index>::to_py() const {
	py::dict d;

#if 1
	const std::vector<ssize_t> shape = {
		static_cast<ssize_t>(m_mapping.size())};
	const uint8_t* const data =
		reinterpret_cast<const uint8_t*>(m_mapping.data());

	d["type"] = py::str("injective");
	d["target"] = PY_ARRAY_MEMBER(HalfEdge, target);
	d["flow"] = PY_ARRAY_MEMBER(HalfEdge, weight.flow);
	d["dist"] = PY_ARRAY_MEMBER(HalfEdge, weight.distance);
#else
	d["type"] = py::str("sparse");
	const size_t n = m_mapping.size();
	for (size_t i = 0; i < n; i++) {
		const auto m = m_mapping[i];
		if (m.target >= 0) {
			sparse.emplace_back(FullEdge{i, m.target, m.weight});
		}
	}
#endif

	return d;
}

template<typename Index>
py::list InjectiveFlow<Index>::py_regions(const Match *p_match, const int p_window_size) const {
	return Flow<Index>::py_regions(p_match, to_edges(), p_window_size);
}

template<typename Index>
py::list InjectiveFlow<Index>::py_omitted(const Match *p_match) const {
	return Flow<Index>::py_omitted(p_match, m_mapping);
}

template<typename Index>
std::vector<typename SparseFlow<Index>::HalfEdge> SparseFlow<Index>::to_injective() const {
	std::vector<HalfEdge> max_flow;
	max_flow.resize(m_source_nodes, HalfEdge{-1, Weight{0.0f, 0.0f}});

	for (const auto &e : m_edges) {
		if (e.weight.flow > max_flow[e.source].weight.flow) {
			max_flow[e.source] = HalfEdge{e.target, e.weight};
		}
	}

	return max_flow;
}

template<typename Index>
py::dict SparseFlow<Index>::to_py() const {
	py::dict d;

	const std::vector<ssize_t> shape = {
		static_cast<ssize_t>(m_edges.size())};
	const uint8_t* const data =
		reinterpret_cast<const uint8_t*>(m_edges.data());

	d["type"] = py::str("sparse");
	d["source"] = PY_ARRAY_MEMBER(Edge, source);
	d["target"] = PY_ARRAY_MEMBER(Edge, target);
	d["flow"] = PY_ARRAY_MEMBER(Edge, weight.flow);
	d["dist"] = PY_ARRAY_MEMBER(Edge, weight.distance);

	return d;
}

template<typename Index>
py::list SparseFlow<Index>::py_regions(const Match *p_match, const int p_window_size) const {
	return Flow<Index>::py_regions(p_match, to_edges(), p_window_size);
}

template<typename Index>
py::list SparseFlow<Index>::py_omitted(const Match *p_match) const {
	return Flow<Index>::py_omitted(p_match, to_injective());
}

template<typename Index>
std::vector<typename DenseFlow<Index>::HalfEdge> DenseFlow<Index>::to_injective() const {
	std::vector<HalfEdge> max_flow;
	max_flow.resize(m_data.shape(0), HalfEdge{-1, Weight{0.0f, 0.0f}});

	const auto indices = xt::argmax(xt::view(m_data, xt::all(), xt::all(), 0), 1);
	PPK_ASSERT(indices.shape(0) == m_data.shape(0));

	for (size_t i = 0; i < indices.size(); i++) {
		const auto j = indices[i];
		const auto f = m_data(i, j, 0);
		if (f > 0.0f) {
			const auto d = m_data(i, j, 1);
			max_flow[i] = HalfEdge{static_cast<Index>(j), Weight{f, d}};
		}
	}

	return max_flow;
}

template<typename Index>
py::dict DenseFlow<Index>::to_py() const {
	py::dict d;

	d["type"] = py::str("dense");
	d["flow"] = xt::pyarray<float>(xt::view(m_data, xt::all(), xt::all(), 0));
	d["dist"] = xt::pyarray<float>(xt::view(m_data, xt::all(), xt::all(), 1));

	return d;
}

template<typename Index>
py::list DenseFlow<Index>::py_regions(const Match *p_match, const int p_window_size) const {
	return Flow<Index>::py_regions(p_match, to_edges(), p_window_size);
}

template<typename Index>
py::list DenseFlow<Index>::py_omitted(const Match *p_match) const {
	return Flow<Index>::py_omitted(p_match, to_injective());
}


template py::list Flow<int16_t>::py_regions(
	const Match *p_match, const std::vector<Edge> &p_edges, const int p_window_size) const;
template py::list Flow<int16_t>::py_omitted(
	const Match *p_match, const std::vector<HalfEdge> &p_edges) const;

template py::dict InjectiveFlow<int16_t>::to_py() const;
template py::list InjectiveFlow<int16_t>::py_regions(const Match *p_match, const int p_window_size) const;
template py::list InjectiveFlow<int16_t>::py_omitted(const Match *p_match) const;

template std::vector<typename SparseFlow<int16_t>::HalfEdge> SparseFlow<int16_t>::to_injective() const;
template py::dict SparseFlow<int16_t>::to_py() const;
template py::list SparseFlow<int16_t>::py_regions(const Match *p_match, const int p_window_size) const;
template py::list SparseFlow<int16_t>::py_omitted(const Match *p_match) const;

template std::vector<typename DenseFlow<int16_t>::HalfEdge> DenseFlow<int16_t>::to_injective() const;
template py::dict DenseFlow<int16_t>::to_py() const;
template py::list DenseFlow<int16_t>::py_regions(const Match *p_match, const int p_window_size) const;
template py::list DenseFlow<int16_t>::py_omitted(const Match *p_match) const;

#endif // __VECTORIAN_FLOW_IMPL_H__
