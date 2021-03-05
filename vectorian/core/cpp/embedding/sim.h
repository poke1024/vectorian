#ifndef __VECTORIAN_EMBEDDING_SIMILARITY_H__
#define __VECTORIAN_EMBEDDING_SIMILARITY_H__

#include "common.h"
#include "embedding/vectors.h"
#include <iostream>

class VocabularyToEmbedding;
class Needle;

class SimilarityMatrixBuilder {
protected:
	const WordVectors &m_embeddings;

public:
	inline SimilarityMatrixBuilder(
		const WordVectors &p_embeddings) : m_embeddings(p_embeddings) {
	}

	virtual ~SimilarityMatrixBuilder() {
	}

	virtual void fill_matrix(
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		xt::xtensor<float, 2> &r_matrix) const = 0;

	void build_similarity_matrix(
		const VocabularyToEmbedding &p_vocabulary_to_embedding,
		const Needle &p_needle,
		xt::xtensor<float, 2> &r_matrix) const;
};

typedef std::shared_ptr<SimilarityMatrixBuilder> SimilarityMatrixBuilderRef;


class WordMetricDef {
public:
	const std::string name;
	const std::string embedding; // e.g. fasttext
	const std::string metric; // e.g. cosine
	const py::dict options;

	SimilarityMatrixBuilderRef instantiate(
		const WordVectors &p_vectors) const;
};


template<typename Distance>
class BuiltinSimilarityMatrixBuilder : public SimilarityMatrixBuilder {
protected:
	Distance m_distance;

public:
	BuiltinSimilarityMatrixBuilder(
		const WordVectors &p_embeddings,
		const Distance p_distance = Distance()) :

		SimilarityMatrixBuilder(p_embeddings),
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
						score = m_distance(m_embeddings, s, t);
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
	filtered_a.reshape({k});
	return filtered_a;
}

class CustomSimilarityMatrixBuilder : public SimilarityMatrixBuilder {
	const py::object m_callback;

public:
	CustomSimilarityMatrixBuilder(
		const WordVectors &p_embeddings,
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

		const py::dict vectors = m_embeddings.to_py();

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


#endif // __VECTORIAN_EMBEDDING_SIMILARITY_H__
