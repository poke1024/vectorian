#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

/*
struct StaticEmbeddingVectors {
	typedef xt::xtensor<float, 2, xt::layout_type::row_major> V;

	V m_unmodified;
	V m_normalized;
	xt::xtensor<float, 1> m_magnitudes;

	inline size_t size() const {
		return m_unmodified.shape(0);
	}

	inline auto unmodified(const size_t p_index) const {
		return xt::view(m_unmodified, p_index, xt::all());
	}

	inline auto normalized(const size_t p_index) const {
		return xt::view(m_normalized, p_index, xt::all());
	}

	inline float magnitude(const size_t p_index) const {
		return m_magnitudes(p_index);
	}

	// specific to static embeddings:

	inline auto &mutable_unmodified() {
		return m_unmodified;
	}

	inline const auto &magnitudes() const {
		return m_magnitudes;
	}

	void free_unmodified() {
		m_unmodified.resize({0, 0});
	}

	void free_normalized() {
		m_normalized.resize({0, 0});
	}

	void update_normalized() {
		compute_magnitudes();

		constexpr float eps = std::numeric_limits<float>::epsilon() * 100.0f;
		const size_t n = m_unmodified.shape(0);
		m_normalized.resize({n, m_unmodified.shape(1)});
		for (size_t i = 0; i < n; i++) {
			const float len = m_magnitudes(i);
			if (len > eps) {
				const auto row = xt::view(m_unmodified, i, xt::all());
				xt::view(m_normalized, i, xt::all()) = row / len;
			} else {
				xt::view(m_normalized, i, xt::all()).fill(0.0f);
			}
		}
	}

	void compute_magnitudes() {
		const size_t n = m_unmodified.shape(0);
		if (m_magnitudes.shape(0) == n) {
			return;
		}
		m_magnitudes.resize({n});

		for (size_t i = 0; i < n; i++) {
			const auto row = xt::view(m_unmodified, i, xt::all());
			m_magnitudes(i) = xt::linalg::norm(row);
		}
	}

	py::dict to_py() const {
		py::dict d;
		d[py::str("unmodified")] = xt::pyarray<float>(m_unmodified);
		d[py::str("normalized")] = xt::pyarray<float>(m_normalized);
		return d;
	}
};*/

#if 0
// ContextualEmbeddingVectors should be implemented in py?

class ContextualEmbeddingVectors {
	xt::xtensor<float, 2> m_vectors;

public:
	/*ContextualEmbeddingVectors(
		const std::string &p_path,
		const size_t p_num_vectors,
		const size_t p_num_dimensions) {

		// https://github.com/xtensor-stack/xtensor-io/pull/18/files#diff-4f602a45ff1e0fd1ebc810d7566a0b98R175
		size_t sz = shape[0] * shape[1];
		const size_t length = sz * sizeof(T);
		void *ptr = mmap::mmap(
			nullptr, length + offset - pa_offset, PROT_READ,
			MAP_PRIVATE, *fd, pa_offset);
		T* t_ptr = reinterpret_cast<T*>(ptr);
		m_vectors = xt::adapt(
			ptr, {p_num_vectors, p_num_dimensions}, xt::no_ownership());
	}*/

	~ContextualEmbeddingVectors() {
	}


	inline auto unmodified(size_t p_index) const {
		return xt::view(m_vectors, p_index, xt::all());
	}

	inline auto normalized(size_t p_index) const {
		const auto v = unmodified(p_index);
		return v / xt::linalg::norm(v);
	}

	inline float magnitude(const size_t p_index) const {
		const auto v = unmodified(p_index);
		return xt::linalg::norm(v);
	}
};

typedef std::shared_ptr<ContextualEmbeddingVectors> ContextualEmbeddingVectorsRef;

class ContextualEmbeddingVectorsFactory {
public:
	ContextualEmbeddingVectorsFactory(const std::string &p_path) {
	}

	ContextualEmbeddingVectorsRef open() {
		return ContextualEmbeddingVectorsRef();
	}
};

typedef std::shared_ptr<ContextualEmbeddingVectorsFactory> ContextualEmbeddingVectorsFactoryRef;

class ContextualSimilarityMatrix {
public:
};

typedef std::shared_ptr<ContextualSimilarityMatrix> ContextualSimilarityMatrixRef;
#endif

class ContextualVectorsContainer {
protected:
	std::unordered_map<std::string, py::object> m_contextual_vectors;

public:
	py::object get_contextual_embedding_vectors(
		const std::string &p_name) const {

		auto it = m_contextual_vectors.find(p_name);
		if (it == m_contextual_vectors.end()) {
			std::ostringstream err;
			err << "could not find embedding " << p_name;
			throw std::runtime_error(err.str());
		}
		return it->second;
	}

};

#endif // __VECTORIAN_WORD_VECTORS_H__
