#ifndef __VECTORIAN_FAST_SCORES_H__
#define __VECTORIAN_FAST_SCORES_H__

#include "common.h"
#include "document.h"
#include "query.h"
#include "metric/static.h"
#include "slice/encoder.h"

template<typename Index>
class StaticEmbeddingSlice {
	const StaticEmbeddingMetric &m_metric;
	const size_t m_slice_id;
	const Token * const s_tokens;
	const Index m_len_s;
	const Token * const t_tokens;
	const Index m_len_t;
	const TokenIdEncoder m_encoder;

public:
	typedef TokenIdEncoder Encoder;

	inline StaticEmbeddingSlice(
		const StaticEmbeddingMetric &metric,
		const size_t slice_id,
		const TokenSpan &s,
		const TokenSpan &t) :

		m_metric(metric),
		m_slice_id(slice_id),
		s_tokens(s.tokens + s.offset),
		m_len_s(s.len),
		t_tokens(t.tokens + t.offset),
		m_len_t(t.len) {
	}

	size_t id() const {
		return m_slice_id;
	}

	inline const TokenIdEncoder &encoder() const {
		return m_encoder;
	}

	inline const Token &s(Index i) const {
		return s_tokens[i];
	}

	inline const Token &t(Index i) const {
		return t_tokens[i];
	}

	inline Index len_s() const {
		return m_len_s;
	}

	inline Index len_t() const {
		return m_len_t;
	}

	inline float similarity(Index i, Index j) const {
		const Token &s = s_tokens[i];
		const auto &sim = m_metric.similarity();
		return sim(m_encoder.to_embedding(s), j);
	}

	inline float magnitude_s(Index i) const {
		const Token &s = s_tokens[i];
		return m_metric.magnitudes()(m_encoder.to_embedding(s));
	}

	inline float magnitude_t(Index i) const {
		const Token &t = t_tokens[i];
		return m_metric.magnitudes()(m_encoder.to_embedding(t));
	}

	inline void assert_has_magnitudes() const {
		m_metric.assert_has_magnitudes();
	}

	inline float max_similarity_for_t(Index i) const {
		return 1.0f;
	}

	inline float max_sum_of_similarities() const {
		return m_len_t;
	}

	inline bool similarity_depends_on_pos() const {
		return false;
	}

	inline float unmodified_similarity(Index i, Index j) const {
		return similarity(i, j);
	}
};

struct TagWeightedOptions {
	float pos_mismatch_penalty;
	float similarity_threshold;
	std::vector<float> t_pos_weights; // by index in t
	float t_pos_weights_sum;
};

template<typename Delegate>
class TagWeightedSlice {
	const Delegate m_delegate;
	const TagWeightedOptions &m_modifiers;

public:
	typedef typename Delegate::Encoder Encoder;

	inline TagWeightedSlice(
		const Delegate &p_delegate,
		const TagWeightedOptions &p_modifiers) :

		m_delegate(p_delegate),
		m_modifiers(p_modifiers) {
	}

	size_t id() const {
		return m_delegate.id();
	}

	inline const typename Delegate::Encoder &encoder() const {
		return m_delegate.encoder();
	}

	inline const Token &s(int i) const {
		return m_delegate.s(i);
	}

	inline const Token &t(int i) const {
		return m_delegate.t(i);
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

	inline float magnitude_s(int i) const {
		return m_delegate.magnitude_s(i);
	}

	inline float magnitude_t(int i) const {
		return m_delegate.magnitude_t(i);
	}

	inline void assert_has_magnitudes() const {
		return m_delegate.assert_has_magnitudes();
	}

	inline float max_similarity_for_t(int i) const {
		return m_modifiers.t_pos_weights[i];
	}

	inline float max_sum_of_similarities() const {
		return m_modifiers.t_pos_weights_sum;
	}

	inline bool similarity_depends_on_pos() const {
		return true;
	}

	inline float unmodified_similarity(int i, int j) const {
		return m_delegate.similarity(i, j);
	}


};

template<typename Slice>
class ReversedSlice {
	const Slice &m_slice;

public:
	inline ReversedSlice(const Slice &slice) :
		m_slice(slice) {
	}

	inline size_t id() const {
		return m_slice.id();
	}

	inline typename Slice::Encoder encoder() const {
		return m_slice.encoder();
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

	inline float magnitude_s(int i) const {
		const auto len_s = m_slice.len_s();
		return m_slice.magnitude_s(len_s - 1 - i);
	}

	inline float magnitude_t(int i) const {
		const auto len_t = m_slice.len_t();
		return m_slice.magnitude_t(len_t - 1 - i);
	}

	inline void assert_has_magnitudes() const {
		m_slice.assert_has_magnitudes();
	}

	inline float max_similarity_for_t(int i) const {
		const auto len_t = m_slice.len_t();
		return m_slice.max_similarity_for_t(len_t - 1 - i);
	}

	inline float max_sum_of_similarities() const {
		return m_slice.max_sum_of_similarities();
	}

	inline bool similarity_depends_on_pos() const {
		return m_slice.similarity_depends_on_pos();
	}

	inline float unmodified_similarity(int u, int v) const {
		const auto len_s = m_slice.len_s();
		const auto len_t = m_slice.len_t();
		return m_slice.unmodified_similarity(len_s - 1 - u, len_t - 1 - v);
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
		const QueryRef &p_query,
		const Delegate &p_delegate,
		const DocumentRef &p_document,
		const TokenFilter &p_filter) :

		m_delegate(p_delegate),
		m_filter(p_filter) {

		const auto &slice_strategy = p_query->slice_strategy();
		m_filtered.resize(p_document->spans(
			slice_strategy.level)->max_len(slice_strategy.window_size));
	}

	inline Slice create_slice(
		const size_t slice_id,
		const TokenSpan &s_span,
		const TokenSpan &t_span) const {

		if (m_filter.all()) {
			return m_delegate.create_slice(
				slice_id, s_span, t_span);

		} else {

		    const Token *s = s_span.tokens + s_span.offset;
		    const auto len_s = s_span.len;

		    Token *new_s = m_filtered.data();
	        PPK_ASSERT(static_cast<size_t>(len_s) <= m_filtered.size());

		    int32_t new_len_s = 0;
	        for (int32_t i = 0; i < len_s; i++) {
	            if (m_filter(s[i])) {
	                new_s[new_len_s++] = s[i];
	            }
	        }

			return m_delegate.create_slice(
				slice_id, TokenSpan{new_s, 0, new_len_s}, t_span);
		}
	}
};

template<typename Make>
class SliceFactory {
	const Make m_make;

public:
	typedef typename std::invoke_result<
		Make,
		const size_t,
		const TokenSpan&,
		const TokenSpan&>::type Slice;

	inline SliceFactory(
		const Make &p_make) :
	    m_make(p_make) {
	}

	inline Slice create_slice(
		const size_t slice_id,
		const TokenSpan &s,
		const TokenSpan &t) const {

		return m_make(slice_id, s, t);
	}
};

#endif // __VECTORIAN_FAST_SCORES_H__
