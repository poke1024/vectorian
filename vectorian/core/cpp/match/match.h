#ifndef __VECTORIAN_MATCH_H__
#define __VECTORIAN_MATCH_H__

#include "common.h"
#include "metric/metric.h"
#include "query.h"
#include "match/matcher.h"
#include "match/region.h"
#include <list>

struct MaximumScore {
	float unmatched;
	float matched;
};

template<typename Index>
class InjectiveFlow;

template<typename Index>
class Flow {
public:
	struct Edge {
		Index source;
		Index target;
		float weight;
	};

	struct HalfEdge {
		Index target;
		float weight;
	};

protected:
	py::list py_regions(const Match *p_match, const std::vector<HalfEdge> &p_edges, const int p_window_size) const;
	py::list py_omitted(const Match *p_match, const std::vector<HalfEdge> &p_edges) const;

public:
	virtual ~Flow() {
	}

	//virtual xt::xtensor<float, 2> to_matrix() const = 0;

	/*virtual py::list py_sparse() const = 0;
	virtual xt::pyarray<float, 2> py_dense() const = 0;*/

	virtual py::dict to_py() const = 0;
	virtual py::list py_regions(const Match *p_match, const int p_window_size) const = 0;
	virtual py::list py_omitted(const Match *p_match) const = 0;
};

template<typename Index>
using FlowRef = std::shared_ptr<Flow<Index>>;

template<typename Index>
class InjectiveFlow : public Flow<Index> {
public:
	typedef typename Flow<Index>::HalfEdge HalfEdge;

private:
	std::vector<HalfEdge> m_mapping;

public:
	inline InjectiveFlow() {
	}

	inline InjectiveFlow(const std::vector<Index> &p_map) {
		m_mapping.reserve(p_map.size());
		for (Index i : p_map) {
			m_mapping.emplace_back(HalfEdge{i, 0.0f});
		}
	}

	inline std::vector<HalfEdge> &mapping() {
		return m_mapping;
	}

	inline void reserve(const Index p_size) {
		m_mapping.reserve(p_size);
	}

	inline void initialize(const Index p_size) {
		m_mapping.clear();
		m_mapping.resize(p_size, HalfEdge{-1, 0.0f});
	}

	inline void set(const Index i, const Index j) {
		m_mapping[i].target = j;
	}

	virtual py::dict to_py() const;
	virtual py::list py_regions(const Match *p_match, const int p_window_size) const;
	virtual py::list py_omitted(const Match *p_match) const;

	template<typename Slice>
	inline MaximumScore max_score(
		const Slice &p_slice) const {

		const size_t n = m_mapping.size();

		float matched_score = 0.0f;
		float unmatched_score = 0.0f;

		for (size_t i = 0; i < n; i++) {
			const float s = p_slice.max_similarity_for_t(i);

			if (m_mapping[i].target < 0) {
				unmatched_score += s;
			} else {
				matched_score += s;
			}
		}

		return MaximumScore{unmatched_score, matched_score};
	}
};

template<typename Index>
using InjectiveFlowRef = std::shared_ptr<InjectiveFlow<Index>>;

template<typename Index>
class SparseFlow : public Flow<Index> {
public:
	typedef typename Flow<Index>::HalfEdge HalfEdge;
	typedef typename Flow<Index>::Edge Edge;

private:
	std::vector<Edge> m_edges;
	size_t m_source_nodes;

	std::vector<HalfEdge> to_injective() const;

public:
	inline SparseFlow() : m_source_nodes(0) {
	}

	inline void initialize(const Index p_size, const int p_degree_hint=2) {
		// p_size is the source size.
		m_edges.reserve(p_size * p_degree_hint);
		m_source_nodes = p_size;
	}

	inline void add(const Index i, const Index j, const float w) {
		m_edges.emplace_back(Edge{i, j, w});
	}

	template<typename Slice>
	inline MaximumScore max_score(
		const Slice &p_slice) const {

		float matched_score = 0.0f;
		const size_t n = p_slice.len_t();
		for (size_t i = 0; i < n; i++) {
			matched_score += p_slice.max_similarity_for_t(i);
		}

		return MaximumScore{0.0f, matched_score};
	}

	virtual py::dict to_py() const;
	virtual py::list py_regions(const Match *p_match, const int p_window_size) const;
	virtual py::list py_omitted(const Match *p_match) const;
};

template<typename Index>
using SparseFlowRef = std::shared_ptr<SparseFlow<Index>>;

template<typename Index>
class DenseFlow : public Flow<Index> {
	xt::xtensor<float, 2> m_matrix;
};

template<typename Index>
using DenseFlowRef = std::shared_ptr<DenseFlow<Index>>;

template<typename Index>
class FlowFactory {
	// foonathan::memory::memory_pool<> m_pool;
	// m_pool(foonathan::memory::list_node_size<Index>::value, 4_KiB)

public:
	InjectiveFlowRef<Index> create_injective(
		const std::vector<Index> &p_match) {
		return std::make_shared<InjectiveFlow<Index>>(p_match);
	}

	InjectiveFlowRef<Index> create_injective() {
		return std::make_shared<InjectiveFlow<Index>>();
	}

	SparseFlowRef<Index> create_sparse() {
		return std::make_shared<SparseFlow<Index>>();
	}
};

template<typename Index>
using FlowFactoryRef = std::shared_ptr<FlowFactory<Index>>;

class MatchDigest {
public:
	DocumentRef document;
	int32_t slice_id;
	FlowRef<int16_t> flow;

	template<template<typename> typename C>
	struct compare;

	inline MatchDigest(
		const DocumentRef &p_document,
		const int32_t p_slice_id,
		const FlowRef<int16_t> &p_flow) :

		document(p_document),
		slice_id(p_slice_id),
		flow(p_flow) {
	}
};

class Match {
private:
	MatcherRef m_matcher;
	const MatchDigest m_digest;
	float m_score; // overall score

public:
	Match(
		const MatcherRef &p_matcher,
		MatchDigest &&p_digest,
		float p_score);

	Match(
		const MatcherRef &p_matcher,
		const DocumentRef &p_document,
		const int32_t p_slice_id,
		const FlowRef<int16_t> &p_flow,
		const float p_score);

	inline const MatcherRef &matcher() const {
		return m_matcher;
	}

	inline const QueryRef &query() const {
		return m_matcher->query();
	}

	inline const MetricRef &metric() const {
		return m_matcher->metric();
	}

	inline const std::string &metric_name() const {
		return metric()->name();
	}

	inline float score() const {
		return m_score;
	}

	inline int32_t slice_id() const {
		return m_digest.slice_id;
	}

	inline const FlowRef<int16_t> &flow() const {
		return m_digest.flow;
	}

	py::dict flow_to_py() const;

	Slice slice() const;

	py::list regions(const int window_size) const;

	py::list omitted() const;

	inline const DocumentRef &document() const {
		return m_digest.document;
	}

	template<template<typename> typename C>
	struct compare_by_score;

	using is_greater = compare_by_score<std::greater>;

	using is_less = compare_by_score<std::less>;

	template<typename Scores>
	void compute_scores(const Scores &p_scores);

	/*void print_scores() const {
		for (auto s : m_scores) {
			printf("similarity: %f, weight: %f\n", s.similarity, s.weight);
		}
	}*/
};

typedef std::shared_ptr<Match> MatchRef;

#endif // __VECTORIAN_MATCH_H__
