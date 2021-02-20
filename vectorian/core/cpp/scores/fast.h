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

template<typename EmbeddingEncoder>
class FastSlice {
private:
	const FastMetricRef m_metric;
	const Token * const s_tokens;
	const int32_t _s_len;
	const Token * const t_tokens;
	const EmbeddingEncoder m_encoder;

public:
	typedef EmbeddingEncoder Encoder;

	inline FastSlice(
		const FastMetricRef &metric,
		const Token * const s_tokens,
		const int32_t s_len,
		const Token * const t_tokens,
		const EmbeddingEncoder &p_encoder) :

		m_metric(metric),
		s_tokens(s_tokens),
		_s_len(s_len),
		t_tokens(t_tokens),
		m_encoder(p_encoder) {
	}

	inline const EmbeddingEncoder &encoder() const {
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

	inline float similarity(int i, int j) const {
		const Token &s = s_tokens[i];
		const auto &sim = m_metric->similarity();
		return sim(m_encoder.to_embedding(s), j);
	}

	inline float weight(int i, int j) const {
		const Token &s = s_tokens[i];
		const Token &t = t_tokens[j];

		// weight based on PennTree POS tag.
		float weight = m_metric->pos_weight(t.tag);

		// difference based on universal POS tag.
		if (s.pos != t.pos) {
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

	template<typename EmbeddingEncoder>
	FastSlice<EmbeddingEncoder> create_slice(
		const size_t p_s_offset,
		const size_t p_s_len,
		const TokenFilter &p_filter,
		const EmbeddingEncoder &p_encoder) const {

		const Token *s_tokens = m_document->tokens()->data();
		const Token *t_tokens = m_query->tokens()->data();

		if (!p_filter.all()) {
		    const Token *s = s_tokens + p_s_offset;
		    Token *new_s = m_filtered.data();
	        PPK_ASSERT(p_s_len <= m_filtered.size());

		    size_t new_s_len = 0;
	        for (size_t i = 0; i < p_s_len; i++) {
	            if (p_filter(s[i])) {
	                new_s[new_s_len++] = s[i];
	            }
	        }

	        return FastSlice<EmbeddingEncoder>(
	            m_metric,
	            new_s,
	            new_s_len,
	            t_tokens,
	            p_encoder);
		}
	    else {
	        return FastSlice<EmbeddingEncoder>(
	            m_metric,
	            s_tokens + p_s_offset,
	            p_s_len,
	            t_tokens,
	            p_encoder);
	    }
	}

	inline bool good() const {
		return true;
	}

	inline int variant() const {
		return 0;
	}
};

#endif // __VECTORIAN_FAST_SCORES_H__
