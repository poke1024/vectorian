#ifndef __VECTORIAN_CONTEXTUAL_EMBEDDING_SLICE_H__
#define __VECTORIAN_CONTEXTUAL_EMBEDDING_SLICE_H__

#include "slice/encoder.h"
#include "embedding/vectors.h"

template<typename Index>
class ContextualEmbeddingSlice {
public:
	typedef Index SliceIndex;
	typedef ContextualEmbeddingTokenIdEncoder Encoder;

private:
	const xt::pytensor<float, 2> &m_matrix;
	const size_t m_slice_id;
	const TokenSpan m_s;
	const TokenSpan m_t;
	const Encoder m_encoder;

public:
	inline ContextualEmbeddingSlice(
		const xt::pytensor<float, 2> &matrix,
		const size_t slice_id,
		const TokenSpan &s,
		const TokenSpan &t) :

		m_matrix(matrix),
		m_slice_id(slice_id),
		m_s(s),
		m_t(t) {

		//std::cout << "ContextualEmbeddingSlice << " << slice_id << ": (" <<
		//	m_s.offset << ", " << (m_s.offset + m_s.len) << ")\n" << std::flush;
	}

	size_t id() const {
		return m_slice_id;
	}

	inline const ContextualEmbeddingTokenIdEncoder &encoder() const {
		return m_encoder;
	}

	inline const Token &s(Index i) const {
		return m_s.tokens[m_s.offset + i];
	}

	inline const Token &t(Index i) const {
		return m_t.tokens[m_t.offset + i];
	}

	inline int32_t len_s() const {
		return m_s.len;
	}

	inline int32_t len_t() const {
		return m_t.len;
	}

	inline Dependency similarity_dependency() const {
		return POSITION;
	}

	inline float similarity(Index i, Index j) const {
		return m_matrix(m_s.offset + i, m_t.offset + j);
	}

	inline float unmodified_similarity(Index i, Index j) const {
		return similarity(i, j);
	}

	inline float magnitude_s(Index i) const {
		return 1.0f; // FIXME m_magnitudes_s(i);
	}

	inline float magnitude_t(Index i) const {
		return 1.0f; // FIXME m_magnitudes_t(i);
	}

	inline void assert_has_magnitudes() const {
		PPK_ASSERT(false);
	}

	inline float max_similarity_for_t(Index i) const {
		return 1.0f;
	}

	inline float max_sum_of_similarities() const {
		return m_t.len;
	}
};

#endif // __VECTORIAN_CONTEXTUAL_EMBEDDING_SLICE_H__
