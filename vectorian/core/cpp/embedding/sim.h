#ifndef __VECTORIAN_EMBEDDING_SIMILARITY_H__
#define __VECTORIAN_EMBEDDING_SIMILARITY_H__

#include "common.h"
#include "embedding/vectors.h"

class EmbeddingSimilarity {
public:
	virtual ~EmbeddingSimilarity() {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		MatrixXf &r_matrix) const = 0;

	virtual void load_percentiles(const std::string &p_path, const std::string &p_name) {
	}
};

typedef std::shared_ptr<EmbeddingSimilarity> EmbeddingSimilarityRef;


template<typename Distance>
class SimilarityMeasure : public EmbeddingSimilarity {
protected:
	Distance m_distance;

public:
	SimilarityMeasure(
		const WordVectors &p_vectors,
		const Distance p_distance = Distance()) : m_distance(p_distance) {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		MatrixXf &r_matrix) const {

		const size_t n = p_a.rows();
		const size_t m = p_b.rows();
		r_matrix.resize(n, m);

		for (size_t i = 0; i < n; i++) { // e.g. for each token in Vocabulary
			const token_t s = p_a[i];

			if (s >= 0) {
				for (size_t j = 0; j < m; j++) { // e.g. for each token in needle

					const token_t t = p_b[j];
					r_matrix(i, j) = (t >= 0) ? m_distance(p_embeddings, s, t) : 0.0f;
				}

			} else { // token in Vocabulary, but not in Embedding

				for (size_t j = 0; j < m; j++) {
					r_matrix(i, j) = 0.0f;
				}
			}
		}
	}

};


class MaximumSimilarityMeasure : public EmbeddingSimilarity {
private:
	std::vector<EmbeddingSimilarityRef> m_measures;

public:
	MaximumSimilarityMeasure(
		const std::vector<EmbeddingSimilarityRef> &p_measures) : m_measures(p_measures) {
	}

	virtual void build_matrix(
		const WordVectors &p_embeddings,
		const TokenIdArray &p_a,
		const TokenIdArray &p_b,
		MatrixXf &r_matrix) const {

		if (m_measures.empty()) {
			r_matrix.setZero();
		} else {
			MatrixXf temp_matrix;
			temp_matrix.resize(r_matrix.rows(), r_matrix.cols());

			for (size_t i = 0; i < m_measures.size(); i++) {
				MatrixXf &target = (i == 0) ? r_matrix : temp_matrix;
				m_measures[i]->build_matrix(
					p_embeddings, p_a, p_b, target);
				if (i > 0) {
					r_matrix = temp_matrix.array().max(r_matrix.array());
				}
			}
		}
	}
};


std::map<std::string, EmbeddingSimilarityRef> create_similarity_measures(
	const std::string &p_name,
	const WordVectors &p_vectors);

#endif // __VECTORIAN_EMBEDDING_SIMILARITY_H__
