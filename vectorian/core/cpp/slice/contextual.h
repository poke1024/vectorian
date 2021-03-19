class ContextualEmbeddingSlice {
	const ContextualEmbeddingVectors &m_s_vectors;
	const ContextualEmbeddingVectors &m_t_vectors;
	const size_t m_slice_id;
	const Token * const s_tokens;
	const int32_t m_offset_s;
	const int32_t m_len_s;
	const Token * const t_tokens;
	const int32_t m_offset_t;
	const int32_t m_len_t;
	const TokenIdEncoder m_encoder;

public:
	inline ContextualEmbeddingSlice(
		const StaticEmbeddingMetric *metric,
		const size_t slice_id,
		const TokenSpan &s,
		const TokenSpan &t) :

		m_metric(metric),
		m_slice_id(slice_id),
		s_tokens(s.tokens),
		m_len_s(s.len),
		t_tokens(t.tokens),
		m_len_t(t.len) {
	}

	size_t id() const {
		return m_slice_id;
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
		m_s_vectors.get_vector(m_offset_s + i)
		m_t_vectors.get_vector(m_offset_t + j)


		const Token &s = s_tokens[i];
		const auto &sim = m_metric->similarity();
		return sim(m_encoder.to_embedding(s), j);
	}

	inline float magnitude_s(int i) const {
		const Token &s = s_tokens[i];
		return m_metric->magnitude_s(m_encoder.to_embedding(s));
	}

	inline float magnitude_t(int i) const {
		return m_metric->magnitude_t(i);
	}

	inline void assert_has_magnitudes() const {
		return true;
	}

	inline float max_similarity_for_t(int i) const {
		return 1.0f;
	}

	inline float max_sum_of_similarities() const {
		return m_len_t;
	}

	inline bool similarity_depends_on_pos() const {
		return false;
	}

	inline float unmodified_similarity(int i, int j) const {
		return similarity(i, j);
	}
};
