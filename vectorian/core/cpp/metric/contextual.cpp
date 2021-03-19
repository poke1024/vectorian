#include "metric/contextual.h"
#include "embedding/contextual.h"
#include "query.h"
#include "document.h"

MatcherFactoryRef ContextualEmbeddingMetric::create_matcher_factory(
	const QueryRef &p_query) {

	py::gil_scoped_acquire acquire;

	// embeddings is a numpy array that can either be dynamically
	// computed or loaded via a cache (loaded via numpy.memmap).

	const auto metric = std::dynamic_pointer_cast<ContextualEmbeddingMetric>(
		shared_from_this());

	const std::string sentence_metric_kind =
		m_options["metric"].cast<py::str>();

	if (sentence_metric_kind == "alignment-isolated") {

#if 0
		const auto matcher_options = create_alignment_matcher_options(metric->alignment_def());
		if (matcher_options.needs_magnitudes) {
			m_needs_magnitudes = true;
		}

		return MatcherFactory::create(
			matcher_options,
			[p_query, metric] (const DocumentRef &p_document, const auto &p_matcher_options) {

				const ContextualEmbeddingVectorsRef t_vectors = p_query->get_contextual_embedding_vectors(
					m_embedding->name());
				const ContextualEmbeddingVectorsRef s_vectors = p_document->get_contextual_embedding_vectors(
					m_embedding->name());

				// compute a n x m matrix, (n: number of tokens in document, m: number of tokens in needle)
				// might offload this to GPU. use this as basis for ContextualEmbeddingSlice.

				const auto sim_matrix = std::make_shared<ContextualSimilarityMatrix>();
				matrix->matrix.resize({s_vectors->size(), t_vectors->size()});
				//xt::xtensor<float, 2> sim_matrix;
				//sim_matrix.resize({s_vectors->size(), t_vectors->size()});

				/*const auto cb = m_embedding->compute_embedding_callback();
				py:array_t embeddings = cb(p_document->py_doc()).cast<py::array_t>();
				PPK_ASSERT(embeddings.shape(0) == p_document->n_tokens());*/

				const SliceFactoryFactory gen_slices([sim_matrix] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

			        return ContextualEmbeddingSlice(sim_matrix, slice_id, s, t);
				});

				return create_alignment_matcher<int16_t>(
					p_query, p_document, metric, metric->alignment_def(), p_matcher_options,
					gen_slices.create_filtered(p_query, p_document, p_query->token_filter()));
			});
#endif

	} /*else if (sentence_metric_kind == "alignment-tag-weighted") {

		TagWeightedOptions options;
		options.t_pos_weights = parse_tag_weights(p_query, m_options["tag_weights"]);
		options.pos_mismatch_penalty = m_options["pos_mismatch_penalty"].cast<float>();
		options.similarity_threshold = m_options["similarity_threshold"].cast<float>();

		float sum = 0.0f;
		for (float w : options.t_pos_weights) {
			sum += w;
		}
		options.t_pos_weights_sum = sum;

		const auto make_tag_weighted_slice = [metric, options] (
			const TokenSpan &s,
			const TokenSpan &t) {

			return TagWeightedSlice(
				StaticEmbeddingSlice(metric, s, t),
				options);
		};

		const SliceFactoryFactory gen(make_tag_weighted_slice);

		return create_alignment_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_document, token_filter));
	}*/ else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}

	throw std::runtime_error("not implemented");
}

const std::string &ContextualEmbeddingMetric::name() const {
	return m_embedding->name();
}
