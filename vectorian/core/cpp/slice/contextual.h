#include "slice/encoder.h"

template<typename VectorSimilarity>
class ContextualEmbeddingSlice {
	const SimilarityMatrix &m_matrix;
	const size_t m_slice_id;
	const TokenSpan m_s;
	const TokenSpan m_t;
	const TokenIdEncoder m_encoder;

public:
	inline ContextualEmbeddingSlice(
		const SimilarityMatrix &matrix,
		const size_t slice_id,
		const TokenSpan &s,
		const TokenSpan &t) :

		m_matrix(matrix),
		m_slice_id(slice_id),
		m_s(s),
		m_t(t) {
	}

	size_t id() const {
		return m_slice_id;
	}

	inline const TokenIdEncoder &encoder() const {
		return m_encoder;
	}

	inline const Token &s(int i) const {
		return m_s.tokens[m_s.offset + i];
	}

	inline const Token &t(int i) const {
		return m_t.tokens[m_t.offset + i];
	}

	inline int32_t len_s() const {
		return m_s.len;
	}

	inline int32_t len_t() const {
		return m_t.len;
	}

	inline float similarity(int i, int j) const {
		return m_matrix(m_s.offset + i, m_t.offset + j);
	}

	inline float magnitude_s(int i) const {
		return m_s_vectors.magnitude(m_s.offset + i);
	}

	inline float magnitude_t(int i) const {
		return m_t_vectors.magnitude(m_t.offset + i);
	}

	inline void assert_has_magnitudes() const {
		return true;
	}

	inline float max_similarity_for_t(int i) const {
		return 1.0f;
	}

	inline float max_sum_of_similarities() const {
		return m_t.len;
	}

	inline bool similarity_depends_on_pos() const {
		return false;
	}

	inline float unmodified_similarity(int i, int j) const {
		return similarity(i, j);
	}
};
