#ifndef __VECTORIAN_MATCH_H__
#define __VECTORIAN_MATCH_H__

#include "common.h"
#include "metric/metric.h"
#include "query.h"
#include "match/matcher.h"
#include <list>

template<typename Index>
class FlowMemoryPool {
	// foonathan::memory::memory_pool<> m_pool;
	// m_pool(foonathan::memory::list_node_size<Index>::value, 4_KiB)
};

template<typename Index>
class Flow {
public:
	virtual ~Flow() {
	}

	virtual const std::vector<Index> &to_map() const = 0;

	virtual xt::xtensor<float, 2> to_matrix() const = 0;
};

template<typename Index>
class OneToOneFlow : public Flow<Index> {
	std::vector<Index> m_map;

public:
	OneToOneFlow(const Index p_source_size) {
		m_map.resize(p_source_size);
	}
};

template<typename Index>
class OneToNFlow : public Flow<Index> {
	std::vector<std::list<Index>> m_map;

public:
	void add(const Index i, const Index j) {
		m_map[i].push_back(j);
	}
};

template<typename Index>
class NToNFlow : public Flow<Index> {
	xt::xtensor<float, 2> m_matrix;
};

class MatchDigest {
public:
	DocumentRef document;
	int32_t slice_id;
	std::vector<int16_t> match;
	xt::xtensor<float, 2> flow;

	template<template<typename> typename C>
	struct compare;

	inline MatchDigest(
		const DocumentRef &p_document,
		const int32_t p_slice_id,
		const std::vector<int16_t> &p_match) :

		document(p_document),
		slice_id(p_slice_id),
		match(p_match) {
	}
};

struct TokenScore {
	float similarity;
	float weight;
};

class Match {
private:
	MatcherRef m_matcher;

	const MatchDigest m_digest;
	float m_score; // overall score
	std::vector<TokenScore> m_scores;

public:
	Match(
		const MatcherRef &p_matcher,
		MatchDigest &&p_digest,
		float p_score);

	Match(
		const MatcherRef &p_matcher,
		const DocumentRef &p_document,
		const int32_t p_slice_id,
		const std::vector<int16_t> &p_match,
		const float p_score);

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

	inline const std::vector<int16_t> &match() const {
		return m_digest.match;
	}

	inline const xt::xtensor<float, 2> flow() const {
		return m_digest.flow;
	}

	py::dict py_assignment() const;

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
	void compute_scores(const Scores &p_scores, int p_len_s, int p_len_t);

	void print_scores() const {
		for (auto s : m_scores) {
			printf("similarity: %f, weight: %f\n", s.similarity, s.weight);
		}
	}
};

typedef std::shared_ptr<Match> MatchRef;

#endif // __VECTORIAN_MATCH_H__
