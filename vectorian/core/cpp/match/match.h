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
class OneToOneFlow;

template<typename Index>
class Flow {
public:
	virtual ~Flow() {
	}

	//virtual xt::xtensor<float, 2> to_matrix() const = 0;

	/*virtual py::list py_sparse() const = 0;
	virtual xt::pyarray<float, 2> py_dense() const = 0;*/

	virtual py::dict to_py() const = 0;
	virtual py::list py_regions(const Match *p_match, const int window_size) const = 0;
	virtual py::list py_omitted(const Match *p_match) const = 0;
};

template<typename Index>
using FlowRef = std::shared_ptr<Flow<Index>>;

template<typename Index>
class OneToOneFlow : public Flow<Index> {
public:
	struct Edge {
		Index target;
		float similarity;
		float weight;
	};

private:
	std::vector<Edge> m_mapping;

public:
	inline OneToOneFlow() {
	}

	inline OneToOneFlow(const std::vector<Index> &p_map) {
		m_mapping.reserve(p_map.size());
		for (Index i : p_map) {
			m_mapping.emplace_back(Edge{i, 0.0f, 0.0f});
		}
	}

	inline std::vector<Edge> &mapping() {
		return m_mapping;
	}

	inline void reserve(const Index p_size) {
		m_mapping.reserve(p_size);
	}

	inline void initialize(const Index p_size) {
		m_mapping.clear();
		m_mapping.resize(p_size, Edge{-1, 0.0f, 0.0f});
	}

	inline void set(const Index i, const Index j) {
		m_mapping[i].target = j;
	}

	virtual py::dict to_py() const {
		py::dict d;

		const std::vector<ssize_t> shape = {
			static_cast<ssize_t>(m_mapping.size())};
		const uint8_t* const data =
			reinterpret_cast<const uint8_t*>(m_mapping.data());

		d["type"] = py::str("1:1");
		d["idx"] = PY_ARRAY_MEMBER(Edge, target);
		d["sim"] = PY_ARRAY_MEMBER(Edge, similarity);
		d["w"] = PY_ARRAY_MEMBER(Edge, weight);

		return d;
	}

	virtual py::list py_regions(const Match *p_match, const int window_size) const;
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
using OneToOneFlowRef = std::shared_ptr<OneToOneFlow<Index>>;

template<typename Index>
class SparseFlow : public Flow<Index> {
	struct Edge {
		Index target;
		float flow;
	};

	std::vector<std::vector<Edge>> m_mapping;

public:
	inline SparseFlow() {
	}

	inline void resize(const Index p_size) {
		m_mapping.resize(p_size);
	}

	inline void add(const Index i, const Index j, const float w) {
		m_mapping[i].emplace_back(Edge{j, w});
	}

	template<typename Slice>
	inline MaximumScore max_score(
		const Slice &p_slice) const {

		const size_t n = m_mapping.size();

		float matched_score = 0.0f;
		float unmatched_score = 0.0f;

		for (size_t i = 0; i < n; i++) {
			const float s = p_slice.max_similarity_for_t(i);

			if (m_mapping[i].empty()) {
				unmatched_score += s;
			} else {
				matched_score += s;
			}
		}

		return MaximumScore{unmatched_score, matched_score};
	}

	virtual py::dict to_py() const {
		return py::list();
	}

	virtual py::list py_regions(const Match *p_match, const int window_size) const {
		return py::list();
	}

	virtual py::list py_omitted(const Match *p_match) const {
		return py::list();
	}
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
	OneToOneFlowRef<Index> create_1_to_1(
		const std::vector<Index> &p_match) {
		return std::make_shared<OneToOneFlow<Index>>(p_match);
	}

	OneToOneFlowRef<Index> create_1_to_1() {
		return std::make_shared<OneToOneFlow<Index>>();
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
