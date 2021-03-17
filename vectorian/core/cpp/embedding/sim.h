#ifndef __VECTORIAN_EMBEDDING_SIMILARITY_H__
#define __VECTORIAN_EMBEDDING_SIMILARITY_H__

#include "common.h"
#include "embedding/vectors.h"
#include <iostream>

class VocabularyToEmbedding;
class Needle;

class SimilarityMatrixBuilder {
public:
	virtual ~SimilarityMatrixBuilder() {
	}

	virtual void fill_matrix(
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		xt::xtensor<float, 2> &r_matrix) const = 0;

	void build_similarity_matrix(
		const Needle &p_needle,
		const VocabularyToEmbedding &p_vocabulary_to_embedding,
		xt::xtensor<float, 2> &r_matrix) const;
};

typedef std::shared_ptr<SimilarityMatrixBuilder> SimilarityMatrixBuilderRef;

template<typename WordVectors>
class SimilarityMatrixBuilderImpl : public SimilarityMatrixBuilder {
protected:
	const WordVectors &m_embeddings;

public:
	inline SimilarityMatrixBuilderImpl(
		const WordVectors &p_embeddings) : m_embeddings(p_embeddings) {
	}
};

class WordMetricDef {
public:
	const std::string name;
	const std::string embedding; // e.g. fasttext
	const std::string metric; // e.g. cosine
	const py::dict options;

	template<typename WordVectors>
	SimilarityMatrixBuilderRef instantiate(
		const WordVectors &p_vectors) const;
};


template<typename WordVectors, typename Distance>
class BuiltinSimilarityMatrixBuilder : public SimilarityMatrixBuilderImpl<WordVectors> {
protected:
	Distance m_distance;

public:
	BuiltinSimilarityMatrixBuilder(
		const WordVectors &p_embeddings,
		const Distance p_distance = Distance()) :

		SimilarityMatrixBuilderImpl<WordVectors>(p_embeddings),
		m_distance(p_distance) {
	}

	virtual void fill_matrix(
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		xt::xtensor<float, 2> &r_matrix) const {

		const size_t n = p_a.shape(0);
		const size_t m = p_b.shape(0);

		/*std::cout << "r_matrix: (" << r_matrix.rows() << ", " << r_matrix.cols() << ")\n";
		std::cout << "i0, n, i0 + n: " << i0 << ", " << n << ", " << (i0 + n) << "\n";
		std::cout << "j0, m, j0 + m: " << j0 << ", " << m << ", " << (j0 + m) << "\n";*/

		PPK_ASSERT(i0 + n <= static_cast<size_t>(r_matrix.shape(0)));
		PPK_ASSERT(j0 + m <= static_cast<size_t>(r_matrix.shape(1)));

		for (size_t i = 0; i < n; i++) { // e.g. for each token in Vocabulary
			const token_t s = p_a[i];
			auto row = xt::view(r_matrix, i + i0, xt::all());

			if (s >= 0) {
				for (size_t j = 0; j < m; j++) { // e.g. for each token in needle

					const token_t t = p_b[j];
					float score;

					if (s == t) {
						score = 1.0f;
					} else if (t >= 0) {
						score = m_distance(this->m_embeddings, s, t);
					} else {
						score = 0.0f;
					}

					row(j + j0) = score;
				}

			} else { // token in Vocabulary, but not in Embedding

				row.fill(0.0f);
			}
		}
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

template<typename WordVectors>
class CustomSimilarityMatrixBuilder : public SimilarityMatrixBuilderImpl<WordVectors> {
	const py::object m_callback;

public:
	CustomSimilarityMatrixBuilder(
		const WordVectors &p_embeddings,
		const py::object p_callback) :

		SimilarityMatrixBuilderImpl<WordVectors>(p_embeddings),
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
};

struct Cosine {
	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		//PPK_ASSERT(p_s >= 0 && p_s < p_vectors.normalized.rows());
		//PPK_ASSERT(p_t >= 0 && p_t < p_vectors.normalized.rows());

		const auto s = xt::view(p_vectors.normalized, p_s, xt::all());
		const auto t = xt::view(p_vectors.normalized, p_t, xt::all());
		return xt::linalg::dot(s, t)();
	}
};

struct ZhuCosine {
	// Zhu et al.

	template<typename WordVectors>
	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		const auto s = xt::view(p_vectors.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vectors.unmodified, p_t, xt::all());

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
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		const auto s = xt::view(p_vectors.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vectors.unmodified, p_t, xt::all());

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
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		const auto s = xt::view(p_vectors.unmodified, p_s, xt::all());
		const auto t = xt::view(p_vectors.unmodified, p_t, xt::all());
		const float d = xt::sum(xt::pow(xt::abs(s - t), m_p))();
		return std::max(0.0f, 1.0f - std::pow(d, 1.0f / m_p) * m_distance_scale);
	}
};

template<typename WordVectors>
SimilarityMatrixBuilderRef WordMetricDef::instantiate(
	const WordVectors &p_vectors) const {

	if (metric == "cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<WordVectors, Cosine>>(p_vectors);
	} if (metric == "zhu-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<WordVectors, ZhuCosine>>(p_vectors);
	} if (metric == "sohangir-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<WordVectors, SohangirCosine>>(p_vectors);
	} else if (metric == "p-norm") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<WordVectors, PNorm>>(
			p_vectors, PNorm(
				options["p"].cast<float>(),
				options["scale"].cast<float>()));

	} else if (metric == "custom") {
		return std::make_shared<CustomSimilarityMatrixBuilder<WordVectors>>(
			p_vectors, options["fn"]);
	} else {
		std::ostringstream err;
		err << "unsupported metric " << metric;
		throw std::runtime_error(err.str());
	}
}

#endif // __VECTORIAN_EMBEDDING_SIMILARITY_H__
