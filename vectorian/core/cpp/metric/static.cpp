#include "metric/static.h"
#include "slice/static.h"
#include "query.h"
#include "match/matcher_impl.h"
#include "metric/alignment.h"
#include "metric/factory.h"

std::vector<float> parse_tag_weights(
	const QueryRef &p_query,
	py::dict tag_weights) {

	std::map<std::string, float> pos_weights;
	for (const auto &pw : tag_weights) {
		pos_weights[pw.first.cast<py::str>()] = pw.second.cast<float>();
	}

	const auto mapped_pos_weights =
		p_query->vocabulary()->mapped_pos_weights(pos_weights);
	const auto t_tokens = p_query->tokens();

	std::vector<float> t_tokens_pos_weights;
	t_tokens_pos_weights.reserve(t_tokens->size());
	for (size_t i = 0; i < t_tokens->size(); i++) {
		const Token &t = t_tokens->at(i);
		const auto w = mapped_pos_weights.find(t.tag);
		float s;
		if (w != mapped_pos_weights.end()) {
			s = w->second;
		} else {
			s = 1.0f;
		}
		t_tokens_pos_weights.push_back(s);
	}

	return t_tokens_pos_weights;
}

MatcherRef StaticEmbeddingMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	py::gil_scoped_acquire acquire;

	const auto metric = std::dynamic_pointer_cast<StaticEmbeddingMetric>(
		shared_from_this());

	const std::string sentence_metric_kind =
		m_options["metric"].cast<py::str>();
	const auto &token_filter = p_query->token_filter();

	if (sentence_metric_kind == "alignment-isolated") {

		const auto make_fast_slice = [metric] (
			const TokenSpan &s,
			const TokenSpan &t) {

	        return StaticEmbeddingSlice(metric.get(), s, t);
		};

		const FactoryGenerator gen(make_fast_slice);

		return create_alignment_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_query, p_document, token_filter));

	} else if (sentence_metric_kind == "alignment-tag-weighted") {

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
				StaticEmbeddingSlice(metric.get(), s, t),
				options);
		};

		const FactoryGenerator gen(make_tag_weighted_slice);

		return create_alignment_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_query, p_document, token_filter));
	} else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}
}

const std::string &StaticEmbeddingMetric::name() const {
	return m_embedding->name();
}
