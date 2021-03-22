#include "result_set.h"
#include "metric/contextual.h"
#include "metric/alignment.h"
#include "embedding/contextual.h"
#include "query.h"
#include "document.h"
#include "metric/factory.h"
#include "slice/contextual.h"

MatcherFactoryRef ContextualEmbeddingMetric::create_matcher_factory(
	const QueryRef &p_query,
	const WordMetricDef &p_word_metric) {

	py::gil_scoped_acquire acquire;

	// embeddings is a numpy array that can either be dynamically
	// computed or loaded via a cache (loaded via numpy.memmap).

	const auto metric = std::dynamic_pointer_cast<ContextualEmbeddingMetric>(
		shared_from_this());

	const std::string sentence_metric_kind =
		m_sent_metric_def["metric"].cast<py::str>();

	if (sentence_metric_kind == "alignment-isolated") {

		bool needs_magnitudes = false;
		const auto matcher_options = create_alignment_matcher_options(metric->alignment_def());
		if (matcher_options.needs_magnitudes) {
			needs_magnitudes = true;
		}

		const py::object vector_metric = p_word_metric.vector_metric;

		return MatcherFactory::create(
			matcher_options,
			[p_query, metric, vector_metric] (
				const DocumentRef &p_document, const auto &p_matcher_options) {

				py::gil_scoped_acquire acquire;

				const HandleRef t_vectors = p_query->vectors_cache().open(
					p_document->get_contextual_embedding_vectors(metric->name())
				);
				const HandleRef s_vectors = p_query->vectors_cache().open(
					p_query->get_contextual_embedding_vectors(metric->name())
				);

				// compute a n x m matrix, (n: number of tokens in document, m: number of tokens in needle)
				// might offload this to GPU. use this as basis for ContextualEmbeddingSlice.

				const auto sim_matrix = std::make_shared<ContextualSimilarityMatrix>();

				sim_matrix->matrix.resize({
					s_vectors->get().attr("size").cast<ssize_t>(),
					t_vectors->get().attr("size").cast<ssize_t>()});

				vector_metric(*s_vectors, *t_vectors, sim_matrix->matrix);

				const SliceFactoryFactory gen_slices([sim_matrix] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

			        return ContextualEmbeddingSlice(sim_matrix->matrix, slice_id, s, t);
				});

				return create_alignment_matcher<int16_t>(
					p_query, p_document, metric, metric->alignment_def(), p_matcher_options,
					gen_slices.create_filtered(p_query, p_document, p_query->token_filter()));
			});

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
}

const std::string &ContextualEmbeddingMetric::name() const {
	return m_embedding->name();
}
