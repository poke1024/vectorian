#include "embedding/sim.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include "query.h"

void SimilarityMatrixBuilder::build_similarity_matrix(
	const QueryRef &p_query,
	xt::xtensor<float, 2> &r_matrix) const {

	//py::gil_scoped_release release;

	const QueryVocabularyRef p_vocabulary = p_query->vocabulary();
	const Needle needle(p_query);

	const size_t vocab_size = p_vocabulary->size();
	const size_t needle_size = static_cast<size_t>(needle.size());
	r_matrix.resize({vocab_size, needle_size});

	const auto &needle_tokens = needle.token_ids();

	size_t offset = 0;
	for (const auto &embedding : m_embeddings) {
		const auto &vectors = embedding->vectors();

		const size_t size = vectors.size();
		PPK_ASSERT(offset + size <= vocab_size);

		for (size_t j = 0; j < needle_size; j++) { // for each token in needle
			const auto t = needle_tokens[j];
			size_t t_rel;
			const auto &t_vectors = pick_vectors(m_embeddings, t, t_rel);

			fill_matrix(vectors, offset, size, t_vectors, t_rel, j, r_matrix);

			if (p_query->debug_hook().has_value()) {
				const auto vec_data = vectors.to_py();
				py::dict data;
				data["s"] = vec_data["normalized"];
				data["t"] = xt::pyarray<float>(t_vectors.normalized(t_rel));
				data["similarity"] = xt::pyarray<float>(
					xt::view(r_matrix, xt::range(offset, offset + size), j));
				(*p_query->debug_hook())("fill_matrix", data);
			}
		}

		offset += size;
	}
	PPK_ASSERT(offset == vocab_size);

	/*p_vocabulary_to_embedding.iterate([&] (const auto &embedding_token_ids, size_t offset) {
		fill_matrix(
			embedding_token_ids,
			p_needle.token_ids(),
			offset,
			0,
			r_matrix);
	});*/

	for (size_t j = 0; j < needle.size(); j++) { // for each token in needle

		// since the j-th needle token is a specific vocabulary token, we always
		// set that specific vocabulary token similarity to 1 (regardless of the
		// embedding distance).
		const auto k = needle_tokens[j];
		if (k >= 0) {
			r_matrix(k, j) = 1.0f;
		}
	}
}
