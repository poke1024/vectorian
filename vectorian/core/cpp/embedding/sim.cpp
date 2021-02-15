#include "embedding/sim.h"

struct CosineSimilarity {
	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		return p_vectors.normalized.row(p_s).dot(p_vectors.normalized.row(p_t));
	}
};

struct SqrtCosine {
	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		float denom = p_vectors.raw.row(p_s).sum() * p_vectors.raw.row(p_t).sum();
		return (p_vectors.raw.row(p_s) * p_vectors.raw.row(p_t)).array().sqrt().sum() / denom;
	}
};

struct PNormSimilarity {
	float m_p;
	float m_distance_scale;

	PNormSimilarity(float p = 2.0f, float scale = 1.0f) : m_p(p), m_distance_scale(scale) {
	}

	inline float operator()(
		const WordVectors &p_vectors,
		token_t p_s,
		token_t p_t) const {

		auto uv = p_vectors.normalized.row(p_s) - p_vectors.normalized.row(p_t);
		float distance = pow(uv.cwiseAbs().array().pow(m_p).sum(), 1.0f / m_p);
		return std::max(0.0f, 1.0f - distance * m_distance_scale);
	}
};

std::map<std::string, EmbeddingSimilarityRef> create_similarity_measures(
	const std::string &p_name,
	const WordVectors &p_vectors) {

	std::map<std::string, EmbeddingSimilarityRef> measures;
	measures["cosine"] = std::make_shared<SimilarityMeasure<CosineSimilarity>>(p_vectors);
	return measures;
}
