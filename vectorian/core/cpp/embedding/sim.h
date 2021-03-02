#ifndef __VECTORIAN_EMBEDDING_SIMILARITY_H__
#define __VECTORIAN_EMBEDDING_SIMILARITY_H__

#include "common.h"
#include "embedding/vectors.h"
#include <iostream>


class EmbeddingSimilarity {
public:
	virtual ~EmbeddingSimilarity() {
	}

	virtual void fill_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		MatrixXf &r_matrix) const = 0;
};

typedef std::shared_ptr<EmbeddingSimilarity> EmbeddingSimilarityRef;


class WordMetricDef {
public:
	const std::string name;
	const std::string embedding; // e.g. fasttext
	const std::string metric; // e.g. cosine
	const py::dict options;

	EmbeddingSimilarityRef instantiate(
		const WordVectors &p_vectors) const;
};


template<typename Distance>
class BuiltinSimilarityMeasure : public EmbeddingSimilarity {
protected:
	Distance m_distance;

public:
	BuiltinSimilarityMeasure(
		const WordVectors &p_vectors,
		const Distance p_distance = Distance()) : m_distance(p_distance) {
	}

	virtual void fill_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		MatrixXf &r_matrix) const {

		const size_t n = p_a.rows();
		const size_t m = p_b.rows();

		/*std::cout << "r_matrix: (" << r_matrix.rows() << ", " << r_matrix.cols() << ")\n";
		std::cout << "i0, n, i0 + n: " << i0 << ", " << n << ", " << (i0 + n) << "\n";
		std::cout << "j0, m, j0 + m: " << j0 << ", " << m << ", " << (j0 + m) << "\n";*/

		PPK_ASSERT(i0 + n <= static_cast<size_t>(r_matrix.rows()));
		PPK_ASSERT(j0 + m <= static_cast<size_t>(r_matrix.cols()));

		for (size_t i = 0; i < n; i++) { // e.g. for each token in Vocabulary
			const token_t s = p_a[i];
			auto row = r_matrix.row(i + i0);

			if (s >= 0) {
				for (size_t j = 0; j < m; j++) { // e.g. for each token in needle

					const token_t t = p_b[j];
					float score;

					if (s == t) {
						score = 1.0f;
					} else if (t >= 0) {
						score = m_distance(p_embeddings, s, t);
					} else {
						score = 0.0f;
					}

					row(j + j0) = score;
				}

			} else { // token in Vocabulary, but not in Embedding

				row.setZero();
			}
		}
	}
};

inline TokenIdArray filter_token_ids(const TokenIdArray &p_a) {
	const size_t n = p_a.rows();
	TokenIdArray filtered_a;
	filtered_a.resize(n);
	size_t k = 0;
	for (size_t i = 0; i < n; i++) {
		const token_t s = p_a[i];
		if (s >= 0) {
			filtered_a[k++] = s;
		}
	}
	filtered_a.resize(k);
	return filtered_a;
}

class CustomSimilarityMeasure : public EmbeddingSimilarity {
	const py::object m_callback;

public:
	CustomSimilarityMeasure(
		const WordVectors &p_vectors,
		const py::object p_callback) : m_callback(p_callback) {
	}

	virtual void fill_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		const size_t i0,
		const size_t j0,
		MatrixXf &r_matrix) const {

		py::gil_scoped_acquire acquire;

		TokenIdArray filtered_a = filter_token_ids(p_a);
		TokenIdArray filtered_b = filter_token_ids(p_b);

		py::dict vectors = p_embeddings.to_py();

		py::array_t<float> output;
		output.resize({filtered_a.rows(), filtered_b.rows()});

		m_callback(
			vectors,
			to_py_array(filtered_a),
			to_py_array(filtered_b),
			output);

		const auto r_output = output.unchecked<2>();

		const size_t n = p_a.rows();
		const size_t m = p_b.rows();

		PPK_ASSERT(i0 + n <= static_cast<size_t>(r_matrix.rows()));
		PPK_ASSERT(j0 + m <= static_cast<size_t>(r_matrix.cols()));

		size_t u = 0;
		for (size_t i = 0; i < n; i++) {
			auto row = r_matrix.row(i + i0);
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
				row.setZero();
			}
		}
	}
};


#endif // __VECTORIAN_EMBEDDING_SIMILARITY_H__
