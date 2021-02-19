#ifndef __VECTORIAN_FAST_SCORES_H__
#define __VECTORIAN_FAST_SCORES_H__

#include "common.h"
#include "metric/fast.h"

class FastSlice {
private:
	const FastMetricRef m_metric;
	const Token * const s_tokens;
	const int32_t _s_len;
	const Token * const t_tokens;

public:
	inline FastSlice(
		const FastMetricRef &metric,
		const Token * const s_tokens,
		const int32_t s_len,
		const Token * const t_tokens) :

		m_metric(metric),
		s_tokens(s_tokens),
		_s_len(s_len),
		t_tokens(t_tokens) {
	}

	inline const Token &s(int i) const {
		return s_tokens[i];
	}

	inline const Token &t(int i) const {
		return t_tokens[i];
	}

	inline int32_t s_len() const {
	    return _s_len;
	}

	inline float similarity(int i, int j) const {
		const Token &s = s_tokens[i];
		const auto &sim = m_metric->similarity();
		return sim(s.id, j);
	}

	inline float weight(int i, int j) const {

		const Token &s = s_tokens[i];
		const Token &t = t_tokens[j];

		// weight based on PennTree POS tag.
		float weight = m_metric->pos_weight(t.tag);

		// difference based on universal POS tag. do not apply
		// if the token is the same, but only POS is different,
		// since often this will an error in the POS tagging.
		if (s.pos != t.pos && s.id != t.id) {
			weight *= 1.0f - m_metric->modifiers().pos_mismatch_penalty;
		}

		return weight;
	}

	inline float score(int i, int j) const {

		const float score = similarity(i, j) * weight(i, j);

		if (score <= m_metric->modifiers().similarity_threshold) {
			return 0.0f;
		} else {
			return score;
		}
	}
};

class FastScores {
	const QueryRef &m_query;
	const DocumentRef &m_document;
	const FastMetricRef m_metric;

	mutable std::vector<Token> m_filtered;

public:
	FastScores(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const FastMetricRef &p_metric);

	FastSlice create_slice(
		size_t p_s_offset,
		size_t p_s_len,
		int p_pos_filter) const;

	inline bool good() const {
		return true;
	}

	inline int variant() const {
		return 0;
	}
};

#endif // __VECTORIAN_FAST_SCORES_H__
