#ifndef __VECTORIAN_FAST_SCORES_H__
#define __VECTORIAN_FAST_SCORES_H__

#include "common.h"
#include "document.h"
#include "query.h"
#include "metric/fast.h"

class TokenIdEncoder {
public:
	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id;
	}
};

class TokenIdPosEncoder {
	const size_t m_npos;

public:
	TokenIdPosEncoder(const size_t p_npos) : m_npos(p_npos) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_npos + p_token.pos;
	}
};

class TokenIdTagEncoder {
	const size_t m_ntag;

public:
	TokenIdTagEncoder(const size_t p_ntag) : m_ntag(p_ntag) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_ntag + p_token.tag;
	}
};

class FastSlice {
private:
	const FastMetricRef m_metric;
	const Token * const s_tokens;
	const int32_t _s_len;
	const Token * const t_tokens;
	const TokenIdEncoder m_encoder;

public:
	typedef TokenIdEncoder Encoder;

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

	inline const TokenIdEncoder &encoder() const {
		return m_encoder;
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

	inline float unmodified_similarity(int i, int j) const {
		const Token &s = s_tokens[i];
		const auto &sim = m_metric->similarity();
		return sim(m_encoder.to_embedding(s), j);
	}

	inline float weight(int i, int j) const {
		const Token &s = s_tokens[i];
		const Token &t = t_tokens[j];

		// weight based on PennTree POS tag.
		float weight = m_metric->pos_weight_for_t(j);

		// difference based on universal POS tag.
		if (s.pos != t.pos) {
			weight *= 1.0f - m_metric->modifiers().pos_mismatch_penalty;
		}

		return weight;
	}

	inline float modified_similarity(int i, int j) const {

		const float score = unmodified_similarity(i, j) * weight(i, j);

		if (score <= m_metric->modifiers().similarity_threshold) {
			return 0.0f;
		} else {
			return score;
		}
	}

	inline bool similarity_depends_on_pos() const {
		return m_metric->similarity_depends_on_pos();
	}
};

template<typename Delegate>
class FilteredSliceFactory {
	const Delegate m_delegate;
	const TokenFilter m_filter;

	mutable std::vector<Token> m_filtered;

public:
	typedef typename Delegate::slice_t slice_t;

	FilteredSliceFactory(
		const Delegate &p_delegate,
		const TokenFilter &p_filter,
		const DocumentRef &p_document) :

		m_delegate(p_delegate),
		m_filter(p_filter) {

       m_filtered.resize(p_document->max_len_s());
	}

	slice_t create_slice(
		const Token *s_tokens,
		const Token *t_tokens,
		const size_t p_len_s,
		const size_t p_len_t) const {

	    const Token *s = s_tokens;
	    Token *new_s = m_filtered.data();
        PPK_ASSERT(p_len_s <= m_filtered.size());

	    size_t new_s_len = 0;
        for (size_t i = 0; i < p_len_s; i++) {
            if (m_filter(s[i])) {
                new_s[new_s_len++] = s[i];
            }
        }

		return m_delegate.create_slice(
			new_s, t_tokens, new_s_len, p_len_t);
	}
};

class FastSliceFactory {
	const FastMetricRef m_metric;

	mutable std::vector<Token> m_filtered;

public:
	typedef FastSlice slice_t;

	FastSliceFactory(
		const FastMetricRef &p_metric) :

	    m_metric(p_metric) {
	}

	FastSlice create_slice(
		const Token *s_tokens,
		const Token *t_tokens,
		const size_t p_len_s,
		const size_t p_len_t) const {

        return FastSlice(
            m_metric,
            s_tokens,
            p_len_s,
            t_tokens);
	}
};

#endif // __VECTORIAN_FAST_SCORES_H__
