#ifndef __VECTORIAN_MATCH_H__
#define __VECTORIAN_MATCH_H__

#include "common.h"
#include "metric/metric.h"
#include "query.h"

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
	const QueryRef m_query;
	const MetricRef m_metric;
	const int16_t m_scores_id;

	const MatchDigest m_digest;
	float m_score; // overall score
	std::vector<TokenScore> m_scores;

	int _pos_filter() const;

public:
	Match(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const int p_scores_id,
		MatchDigest &&p_digest,
		float p_score);

	inline float score() const {
		return m_score;
	}

	inline int32_t sentence_id() const {
		return m_digest.sentence_id;
	}

	inline const std::vector<int16_t> &match() const {
		return m_digest.match;
	}

	const Sentence &sentence() const;

	py::tuple location() const;

	py::list regions() const;

	py::list omitted() const;

	const std::string &metric() const {
		return m_metric->name();
	}

	inline const int scores_variant_id() const {
		return m_scores_id;
	}

	inline const DocumentRef &document() const {
		return m_digest.document;
	}

	template<template<typename> typename C>
	struct compare_by_score;

	using is_worse = compare_by_score<std::greater>;

	using is_better = compare_by_score<std::less>;

	template<typename Scores>
	void compute_scores(const Scores &p_scores, int p_len_s);

	void print_scores() const {
		for (auto s : m_scores) {
			printf("similarity: %f, weight: %f\n", s.similarity, s.weight);
		}
	}
};

typedef std::shared_ptr<Match> MatchRef;

#endif // __VECTORIAN_MATCH_H__
