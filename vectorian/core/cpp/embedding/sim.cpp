#include "embedding/sim.h"
#include "embedding/static.h"

struct Cosine {
	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		//PPK_ASSERT(p_s >= 0 && p_s < p_vectors.normalized.rows());
		//PPK_ASSERT(p_t >= 0 && p_t < p_vectors.normalized.rows());

		auto s = xt::view(p_vectors.normalized, p_s, xt::all());
		auto t = xt::view(p_vectors.normalized, p_t, xt::all());
		return xt::linalg::dot(s, t)();
	}
};

struct ZhuCosine { // Zhu et al.
	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		/*const float num = (p_vectors.normalized.row(p_s) * p_vectors.normalized.row(p_t)).array().sqrt().sum();
		const float denom = p_vectors.normalized.row(p_s).sum() * p_vectors.normalized.row(p_t).sum();
		return num / denom;*/
		return 0.0f; // FIXME
	}
};

struct SohangirCosine { // Sohangir & Wang
	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		/*const float num = (p_vectors.unmodified.row(p_s) * p_vectors.unmodified.row(p_t)).array().sqrt().sum();
		const float denom = std::sqrt(p_vectors.normalized.row(p_s).sum()) * std::sqrt(p_vectors.normalized.row(p_t).sum());
		return num / denom;*/
		return 0.0f; // FIXME
	}
};

struct PNorm {
	const float m_p;
	const float m_distance_scale;

	inline PNorm(float p = 2.0f, float scale = 1.0f) : m_p(p), m_distance_scale(scale) {
	}

	inline float operator()(
		const WordVectors &p_vectors,
		const token_t p_s,
		const token_t p_t) const {

		const auto s = xt::view(p_vectors.normalized, p_s, xt::all());
		const auto t = xt::view(p_vectors.normalized, p_t, xt::all());
		const float d = xt::sum(xt::pow(xt::abs(s - t), m_p))();
		return std::max(0.0f, 1.0f - std::pow(d, 1.0f / m_p) * m_distance_scale);
	}
};

SimilarityMatrixBuilderRef WordMetricDef::instantiate(
	const WordVectors &p_vectors) const {

	if (metric == "cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<Cosine>>(p_vectors);
	} if (metric == "zhu-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<ZhuCosine>>(p_vectors);
	} if (metric == "sohangir-cosine") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<SohangirCosine>>(p_vectors);
	} else if (metric == "p-norm") {
		return std::make_shared<BuiltinSimilarityMatrixBuilder<PNorm>>(
			p_vectors, PNorm(
				options["p"].cast<float>(),
				options["scale"].cast<float>()));

	} else if (metric == "custom") {
		return std::make_shared<CustomSimilarityMatrixBuilder>(
			p_vectors, options["fn"]);
	} else {
		std::ostringstream err;
		err << "unsupported metric " << metric;
		throw std::runtime_error(err.str());
	}
}

void SimilarityMatrixBuilder::build_similarity_matrix(
	const VocabularyToEmbedding &p_vocabulary_to_embedding,
	const Needle &p_needle,
	xt::xtensor<float, 2> &r_matrix) const {

	py::gil_scoped_release release;

	const size_t vocab_size = p_vocabulary_to_embedding.size();
	//std::cout << "resizing matrix " << vocab_size << " x " << needle_embedding_token_ids.rows() << "\n";
	r_matrix.resize({vocab_size, static_cast<size_t>(p_needle.embedding_token_ids().rows())});

	p_vocabulary_to_embedding.iterate([&] (const auto &embedding_token_ids, size_t offset) {
		fill_matrix(
			embedding_token_ids,
			p_needle.embedding_token_ids(),
			offset,
			0,
			r_matrix);
	});

	for (size_t j = 0; j < p_needle.size(); j++) { // for each token in needle

		// since the j-th needle token is a specific vocabulary token, we always
		// set that specific vocabulary token similarity to 1 (regardless of the
		// embedding distance).
		const auto k = p_needle.vocabulary_token_ids()[j];
		if (k >= 0) {
			r_matrix(k, j) = 1.0f;
		}
	}

}
