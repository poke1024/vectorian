#include "metric/static.h"
#include "metric/contextual.h"
#include "slice/static.h"
#include "slice/contextual.h"
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

// --------------------------------------------------------------------------------

class StaticEmbeddingMatcherFactory : public MinimalMatcherFactory {
public:
    virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document,
		const MatcherOptions &p_matcher_options) const {

		const auto matrix = std::static_pointer_cast<StaticEmbeddingMetric>(p_metric)->matrix();

		return make_matcher(p_query, p_metric, p_document, p_matcher_options, [matrix] (
			const size_t slice_id,
			const TokenSpan &s,
			const TokenSpan &t) {

	        return StaticEmbeddingSlice<Index>(
	            *matrix.get(), slice_id, s, t);
		});
	};
};

class TagWeightedStaticEmbeddingMatcherFactory : public MinimalMatcherFactory {
	const TagWeightedOptions m_options;

public:
	TagWeightedStaticEmbeddingMatcherFactory(
		const TagWeightedOptions &p_options) :

		m_options(p_options) {
	}

    virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document,
		const MatcherOptions &p_matcher_options) const {

		const auto matrix = std::static_pointer_cast<StaticEmbeddingMetric>(p_metric)->matrix();
		const auto options = m_options;

		return make_matcher(p_query, p_metric, p_document, p_matcher_options, [matrix, options] (
			const size_t slice_id,
			const TokenSpan &s,
			const TokenSpan &t) {

			return TagWeightedSlice(
				StaticEmbeddingSlice<Index>(*matrix.get(), slice_id, s, t),
				options);
		});
	};
};


class ContextualEmbeddingMatcherFactory : public MinimalMatcherFactory {
public:
    virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const MetricRef &p_metric,
		const DocumentRef &p_document,
		const MatcherOptions &p_matcher_options) const {

		const auto metric = std::static_pointer_cast<ContextualEmbeddingMetric>(p_metric);
		const SimilarityMatrixRef matrix = metric->matrix_factory()->create(CONTEXTUAL, p_document);

		return make_matcher(p_query, p_metric, p_document, p_matcher_options, [matrix] (
			const size_t slice_id,
			const TokenSpan &s,
			const TokenSpan &t) {

	        return ContextualEmbeddingSlice<Index>(
	            matrix->m_similarity, slice_id, s, t);
		});
	};
};

MatcherFactoryRef create_matcher_factory(
	const QueryRef &p_query,
	const py::dict &sent_metric_def) {

	py::gil_scoped_acquire acquire;

	const std::string sentence_metric_kind =
		sent_metric_def["metric"].cast<py::str>();

	const py::dict alignment_def = sent_metric_def["alignment"].cast<py::dict>();;
	const auto matcher_options = create_alignment_matcher_options(alignment_def);

	if (sentence_metric_kind == "alignment-isolated") {

		return std::make_shared<MatcherFactory>(
			std::make_shared<StaticEmbeddingMatcherFactory>(),
			std::make_shared<ContextualEmbeddingMatcherFactory>(),
			matcher_options);

	} else if (sentence_metric_kind == "alignment-tag-weighted") {

		TagWeightedOptions options;
		options.t_pos_weights = parse_tag_weights(p_query, sent_metric_def["tag_weights"]);
		options.pos_mismatch_penalty = sent_metric_def["pos_mismatch_penalty"].cast<float>();
		options.similarity_threshold = sent_metric_def["similarity_threshold"].cast<float>();

		float sum = 0.0f;
		for (float w : options.t_pos_weights) {
			sum += w;
		}
		options.t_pos_weights_sum = sum;

		return std::make_shared<MatcherFactory>(
			std::make_shared<TagWeightedStaticEmbeddingMatcherFactory>(options),
			MinimalMatcherFactoryRef(), // not supported right now.
			matcher_options);

	} else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}
}
