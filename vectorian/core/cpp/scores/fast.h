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
	inline TokenIdPosEncoder(const size_t p_npos) : m_npos(p_npos) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_npos + p_token.pos;
	}
};

class TokenIdTagEncoder {
	const size_t m_ntag;

public:
	inline TokenIdTagEncoder(const size_t p_ntag) : m_ntag(p_ntag) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_ntag + p_token.tag;
	}
};

class FastSlice {
	const FastMetricRef m_metric;
	const Token * const s_tokens;
	const int32_t m_len_s;
	const Token * const t_tokens;
	const int32_t m_len_t;
	const TokenIdEncoder m_encoder;

public:
	typedef TokenIdEncoder Encoder;

	inline FastSlice(
		const FastMetricRef &metric,
		const TokenSpan &s,
		const TokenSpan &t) :

		m_metric(metric),
		s_tokens(s.tokens),
		m_len_s(s.len),
		t_tokens(t.tokens),
		m_len_t(t.len) {
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

	inline int32_t len_s() const {
		return m_len_s;
	}

	inline int32_t len_t() const {
		return m_len_t;
	}

	inline float similarity(int i, int j) const {
		const Token &s = s_tokens[i];
		const auto &sim = m_metric->similarity();
		return sim(m_encoder.to_embedding(s), j);
	}

	inline float unmodified_similarity(int i, int j) const {
		return similarity(i, j);
	}

	inline bool similarity_depends_on_pos() const {
		return m_metric->similarity_depends_on_pos();
	}
};

template<typename Delegate>
class TagWeightedSlice {
	const Delegate m_delegate;
	const MetricModifiers &m_modifiers;

public:
	inline TagWeightedSlice(
		const Delegate &p_delegate,
		const MetricModifiers &p_modifiers) :

		m_delegate(p_delegate),
		m_modifiers(p_modifiers) {
	}

	inline const typename Delegate::Encoder &encoder() const {
		return m_delegate.encoder();
	}

	inline const Token &s(int i) const {
		return m_delegate.s[i];
	}

	inline const Token &t(int i) const {
		return m_delegate.t[i];
	}

	inline int32_t len_s() const {
		return m_delegate.len_s();
	}

	inline int32_t len_t() const {
		return m_delegate.len_t();
	}

	inline float weight(int i, int j) const {
		const Token &s = m_delegate.s(i);
		const Token &t = m_delegate.t(j);

		// weight based on PennTree POS tag.
		float weight = m_modifiers.t_pos_weights[j];

		// difference based on universal POS tag.
		if (s.pos != t.pos) {
			weight *= 1.0f - m_modifiers.pos_mismatch_penalty;
		}

		return weight;
	}

	inline float similarity(int i, int j) const {
		const float score = m_delegate.similarity(i, j) * weight(i, j);

		if (score <= m_modifiers.similarity_threshold) {
			return 0.0f;
		} else {
			return score;
		}
	}

	inline float unmodified_similarity(int i, int j) const {
		return m_delegate.similarity(i, j)(i, j);
	}
};

template<typename Slice>
class ReversedSlice {
	const Slice &m_slice;

public:
	inline ReversedSlice(const Slice &slice) :
		m_slice(slice) {
	}

	inline const Token &s(int i) const {
		const auto len_s = m_slice.len_s();
		return m_slice.s(len_s - 1 - i);
	}

	inline const Token &t(int i) const {
		const auto len_t = m_slice.len_t();
		return m_slice.t(len_t - 1 - i);
	}

	inline int len_s() const {
	    return m_slice.len_s();
	}

	inline int len_t() const {
	    return m_slice.len_t();
	}

	inline float similarity(int u, int v) const {
		const auto len_s = m_slice.len_s();
		const auto len_t = m_slice.len_t();
		return m_slice.similarity(len_s - 1 - u, len_t - 1 - v);
	}

	inline typename Slice::Encoder encoder() const {
		return m_slice.encoder();
	}

	inline bool similarity_depends_on_pos() const {
		return m_slice.similarity_depends_on_pos();
	}
};

template<typename Delegate>
class FilteredSliceFactory {
	const Delegate m_delegate;
	const TokenFilter m_filter;

	mutable std::vector<Token> m_filtered;

public:
	typedef typename Delegate::Slice Slice;

	inline FilteredSliceFactory(
		const Delegate &p_delegate,
		const DocumentRef &p_document,
		const TokenFilter &p_filter) :

		m_delegate(p_delegate),
		m_filter(p_filter) {

		m_filtered.resize(p_document->max_len_s());
	}

	inline Slice create_slice(
		const TokenSpan &s_span,
		const TokenSpan &t_span) const {

	    const Token *s = s_span.tokens;
	    const auto len_s = s_span.len;

	    Token *new_s = m_filtered.data();
        PPK_ASSERT(len_s <= m_filtered.size());

	    int32_t new_len_s = 0;
        for (int32_t i = 0; i < len_s; i++) {
            if (m_filter(s[i])) {
                new_s[new_len_s++] = s[i];
            }
        }

		return m_delegate.create_slice(
			TokenSpan{new_s, new_len_s}, t_span);
	}
};

template<typename Make>
class SliceFactory {
	const Make m_make;

public:
	typedef typename std::invoke_result<
		Make,
		const TokenSpan&,
		const TokenSpan&>::type Slice;

	inline SliceFactory(
		const Make &p_make) :
	    m_make(p_make) {
	}

	inline Slice create_slice(
		const TokenSpan &s,
		const TokenSpan &t) const {

		return m_make(s, t);
	}
};

#endif // __VECTORIAN_FAST_SCORES_H__
