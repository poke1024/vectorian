#include "embedding/sim.h"
#include "embedding/static.h"

void SimilarityMatrixBuilder::build_similarity_matrix(
	const Needle &p_needle,
	const VocabularyToEmbedding &p_vocabulary_to_embedding,
	xt::xtensor<float, 2> &r_matrix) const {

	py::gil_scoped_release release;

	const size_t vocab_size = p_vocabulary_to_embedding.size();
	//std::cout << "resizing matrix " << vocab_size << " x " << needle_embedding_token_ids.rows() << "\n";
	r_matrix.resize({vocab_size, static_cast<size_t>(p_needle.embedding_token_ids().shape(0))});

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
