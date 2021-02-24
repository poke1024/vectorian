#include "metric/contextual.h"
#include "embedding/contextual.h"
#include "query.h"
#include "document.h"

MatcherRef ContextualEmbeddingMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	py::gil_scoped_acquire acquire;

	/*const auto cb = m_embedding->compute_embedding_callback();
	py:array_t embeddings = cb(p_document->unique_id()).cast<py::array_t>();
	PPK_ASSERT(embeddings.shape(0) == p_document->n_tokens());*/

	// embeddings is a numpy array that can either be dynamically
	// computed or loaded via a cache (loaded via numpy.memmap).

	const auto metric = std::dynamic_pointer_cast<ContextualEmbeddingMetric>(
		shared_from_this());

	const std::string sentence_metric_kind =
		m_options["metric"].cast<py::str>();
	//const auto &token_filter = p_query->token_filter();

	if (sentence_metric_kind == "alignment-isolated") {

#if 0
		const auto make_contextual_slice = [metric] (
			const TokenSpan &s,
			const TokenSpan &t) {

			py::gil_scoped_acquire acquire;

	        return ContextualEmbeddingSlice(metric, embeddings, s, t);
		};

		const FactoryGenerator gen(make_contextual_slice);

		return create_alignment_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_document, token_filter));
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

		const FactoryGenerator gen(make_tag_weighted_slice);

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
