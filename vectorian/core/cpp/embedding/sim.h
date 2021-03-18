#ifndef __VECTORIAN_EMBEDDING_SIMILARITY_H__
#define __VECTORIAN_EMBEDDING_SIMILARITY_H__

#include "common.h"
#include "embedding/vectors.h"
#include "embedding/static.h"
#include <iostream>

//class StaticEmbedding;
//typedef std::shared_ptr<StaticEmbedding> StaticEmbeddingRef;

class Needle;

class SimilarityMatrixBuilder {
protected:
	const std::vector<StaticEmbeddingRef> m_embeddings;

	virtual void fill_matrix(
		const WordVectors &p_s_vectors,
		const size_t p_offset,
		const size_t p_size,
		const WordVectors &p_t_vectors,
		const size_t p_t_index,
		const size_t p_column,
		xt::xtensor<float, 2> &r_matrix) const = 0;

public:
	inline SimilarityMatrixBuilder(
		const std::vector<StaticEmbeddingRef> &p_embeddings) :
		m_embeddings(p_embeddings) {
	}

	virtual ~SimilarityMatrixBuilder() {
	}

	void build_similarity_matrix(
		const QueryRef &p_query,
		xt::xtensor<float, 2> &r_matrix) const;


};

typedef std::shared_ptr<SimilarityMatrixBuilder> SimilarityMatrixBuilderRef;

template<typename Distance>
class BuiltinSimilarityMatrixBuilder : public SimilarityMatrixBuilder {
protected:
	Distance m_distance;

	virtual void fill_matrix(
		const WordVectors &p_s_vectors,
		const size_t p_offset,
		const size_t p_size,
		const WordVectors &p_t_vectors,
		const size_t p_t_index,
		const size_t p_column,
		xt::xtensor<float, 2> &r_matrix) const {

		for (size_t i = 0; i < p_size; i++) { // for each token in vocabulary
			r_matrix(i + p_offset, p_column) = m_distance(
				p_s_vectors, i, p_t_vectors, p_t_index);
		}
	}

public:
	BuiltinSimilarityMatrixBuilder(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const Distance p_distance = Distance()) :

		SimilarityMatrixBuilder(p_embeddings),
		m_distance(p_distance) {
	}
};

inline TokenIdArray filter_token_ids(const TokenIdArray &p_a) {
	const size_t n = p_a.shape(0);
	TokenIdArray filtered_a;
	filtered_a.resize({n});
	size_t k = 0;
	for (size_t i = 0; i < n; i++) {
		const token_t s = p_a[i];
		if (s >= 0) {
			filtered_a[k++] = s;
		}
	}
	return xt::view(filtered_a, xt::range(0, k));
}

/*class CustomSimilarityMatrixBuilder : public SimilarityMatrixBuilder {
	const py::object m_callback;

public:
	CustomSimilarityMatrixBuilder(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const py::object p_callback) :

		SimilarityMatrixBuilder(p_embeddings),
		m_callback(p_callback) {
	}

	virtual void fill_matrix(
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		xt::xtensor<float, 2> &r_matrix) const {

		py::gil_scoped_acquire acquire;

		TokenIdArray filtered_a = filter_token_ids(p_a);
		TokenIdArray filtered_b = filter_token_ids(p_b);

		const py::dict vectors = this->m_embeddings.to_py();

		py::array_t<float> output;
		output.resize({filtered_a.shape(0), filtered_b.shape(0)});

		m_callback(
			vectors,
			xt::pyarray<token_t>(filtered_a),
			xt::pyarray<token_t>(filtered_b),
			output);

		const auto r_output = output.unchecked<2>();

		const size_t n = p_a.shape(0);
		const size_t m = p_b.shape(0);

		PPK_ASSERT(i0 + n <= static_cast<size_t>(r_matrix.shape(0)));
		PPK_ASSERT(j0 + m <= static_cast<size_t>(r_matrix.shape(1)));

		size_t u = 0;
		for (size_t i = 0; i < n; i++) {
			auto row = xt::view(r_matrix, i + i0, xt::all());
			const token_t s = p_a[i];

			if (s >= 0) {
				size_t v = 0;
				for (size_t j = 0; j < m; j++) {
					const token_t t = p_b[j];
					if (t >= 0) {
						row(j + j0) = r_output(u, v++);
					} else {
						row(j + j0) = 0.0f;
					}
				}

				u++;
			} else {
				row.fill(0.0f);
			}
		}
	}
};*/

struct Cosine {
	/*template<typename WordVectors, typename Selector>
	inline auto rows(
		const WordVectors &p_vs,
		const Selector p_s) const {
		return xt::view(p_vs.normalized, p_s, xt::all());
	}*/

	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vs,
		const token_t p_s,
		const WordVectors &p_vt,
		const token_t p_t) const {

		const auto s = xt::view(p_vs.normalized, p_s, xt::all());
		const auto t = xt::view(p_vt.normalized, p_t, xt::all());
		return xt::linalg::dot(s, t)();
	}
};

struct ZhuCosine {
	// Zhu et al.

	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vs,
		const token_t p_s,
		const WordVectors &p_vt,
		const token_t p_t) const {

		const auto s = xt::view(p_vs.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vt.unmodified, p_t, xt::all());

		const float num = xt::sum(xt::sqrt(s * t))();
		const float denom = xt::sum(s)() * xt::sum(t)();
		return num / denom;

	}
};

struct SohangirCosine {
	/*
	Sohangir, Sahar, and Dingding Wang. “Improved Sqrt-Cosine Similarity Measurement.”
	Journal of Big Data, vol. 4, no. 1, Dec. 2017, p. 25. DOI.org (Crossref), doi:10.1186/s40537-017-0083-6.
	*/

	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vs,
		const token_t p_s,
		const WordVectors &p_vt,
		const token_t p_t) const {

		const auto s = xt::view(p_vs.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vt.unmodified, p_t, xt::all());

		const float num = xt::sum(xt::sqrt(s * t))();
		const float denom = std::sqrt(xt::sum(s)()) * std::sqrt(xt::sum(t)());
		return num / denom;
	}
};

struct PNorm {
	const float m_p;
	const float m_distance_scale;

	inline PNorm(float p = 2.0f, float scale = 1.0f) : m_p(p), m_distance_scale(scale) {
	}

	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vs,
		const token_t p_s,
		const WordVectors &p_vt,
		const token_t p_t) const {

		const auto s = xt::view(p_vs.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vt.unmodified, p_t, xt::all());
		const float d = xt::sum(xt::pow(xt::abs(s - t), m_p))();
		return std::max(0.0f, 1.0f - std::pow(d, 1.0f / m_p) * m_distance_scale);
	}
};

inline SimilarityMatrixBuilderRef WordMetricDef::instantiate(
	const std::vector<StaticEmbeddingRef> &p_embeddings) const {

	if (metric == "cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<Cosine>>(p_embeddings);
	} if (metric == "zhu-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<ZhuCosine>>(p_embeddings);
	} if (metric == "sohangir-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<SohangirCosine>>(p_embeddings);
	} else if (metric == "p-norm") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<PNorm>>(
			p_embeddings, PNorm(
				options["p"].cast<float>(),
				options["scale"].cast<float>()));

	/*} else if (metric == "custom") {
		return std::make_shared<CustomSimilarityMatrixBuilder<WordVectors>>(
			p_embeddings, options["fn"]);
	*/
	} else {
		std::ostringstream err;
		err << "unsupported metric " << metric;
		throw std::runtime_error(err.str());
	}
}

#endif // __VECTORIAN_EMBEDDING_SIMILARITY_H__
