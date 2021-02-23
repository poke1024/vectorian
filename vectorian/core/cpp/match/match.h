#ifndef __VECTORIAN_MATCH_H__
#define __VECTORIAN_MATCH_H__

#include "common.h"
#include "metric/metric.h"
#include "query.h"
#include "match/matcher.h"

class MatchDigest {
public:
	DocumentRef document;
	int32_t sentence_id;
	std::vector<int16_t> match;

	template<template<typename> typename C>
	struct compare;

	inline MatchDigest(
		DocumentRef p_document,
		int32_t p_sentence_id,
		const std::vector<int16_t> &p_match) :

		document(p_document),
		sentence_id(p_sentence_id),
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
	const int16_t m_scores_id;

	const MatchDigest m_digest;
	float m_score; // overall score
	std::vector<TokenScore> m_scores;

public:
	Match(
		const MatcherRef &p_matcher,
		const int p_scores_id,
		MatchDigest &&p_digest,
		float p_score);

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

	inline int32_t sentence_id() const {
		return m_digest.sentence_id;
	}

	inline const std::vector<int16_t> &match() const {
		return m_digest.match;
	}

	py::dict py_assignment() const;

	const Sentence &sentence() const;

	py::list regions() const;

	py::list omitted() const;

	inline const int scores_variant_id() const {
		return m_scores_id;
	}

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
