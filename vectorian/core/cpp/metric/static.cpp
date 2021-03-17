#include "metric/static.h"
#include "slice/static.h"
#include "query.h"
#include "match/matcher_impl.h"
#include "metric/alignment.h"
#include "metric/factory.h"

void StaticEmbeddingMetric::initialize(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const VocabularyToEmbedding &p_vocabulary_to_embedding) {

	const auto builder = p_metric.instantiate(
		m_embedding->embeddings());

	const Needle needle(p_query, p_vocabulary_to_embedding);

	builder->build_similarity_matrix(
		needle,
		p_vocabulary_to_embedding,
		m_similarity);

	if (m_options.contains("similarity_falloff")) {
		const float similarity_falloff = m_options["similarity_falloff"].cast<float>();
		m_similarity = xt::pow(m_similarity, similarity_falloff);
	}

	//std::cout << "has debug hook " << p_query->debug_hook().has_value() << "\n";
	if (p_query->debug_hook().has_value()) {
		auto gen_labels = py::cpp_function([&] () {
			const auto &vocab = p_query->vocabulary();

			py::list row_tokens;
			p_vocabulary_to_embedding.iterate([&] (
				const auto &ids, const size_t offset) {
				for (size_t i = 0; i < ids.size(); i++) {
					row_tokens.append(vocab->id_to_token(ids[i]));
				}
			});
			py::list col_tokens;
			for (const auto &t : *p_query->tokens()) {
				col_tokens.append(vocab->id_to_token(t.id));
			}

			py::dict labels;
			labels["rows"] = row_tokens;
			labels["columns"] = col_tokens;
			return labels;
		});

		py::dict data;
		data["matrix"] = xt::pyarray<float>(m_similarity);
		data["labels"] = gen_labels;

		(*p_query->debug_hook())("similarity_matrix", data);
	}

	m_matcher_factory = create_matcher_factory(p_query);

	if (m_needs_magnitudes) { // set in create_matcher_factory
		compute_magnitudes(
			m_embedding->embeddings(),
			p_vocabulary_to_embedding,
			needle);
	}
}

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

MatcherFactoryRef StaticEmbeddingMetric::create_matcher_factory(
	const QueryRef &p_query) {

	py::gil_scoped_acquire acquire;

	const auto metric = std::dynamic_pointer_cast<StaticEmbeddingMetric>(
		shared_from_this());

	const std::string sentence_metric_kind =
		m_options["metric"].cast<py::str>();

	if (sentence_metric_kind == "alignment-isolated") {

		const auto matcher_options = create_alignment_matcher_options(metric->alignment_def());
		if (matcher_options.needs_magnitudes) {
			m_needs_magnitudes = true;
		}

		return MatcherFactory::create(
			matcher_options,
			[p_query, metric] (const DocumentRef &p_document, const auto &p_matcher_options) {
				const auto make_fast_slice = [metric] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

			        return StaticEmbeddingSlice(metric.get(), slice_id, s, t);
				};

				const FactoryGenerator gen(make_fast_slice);
				const auto &token_filter = p_query->token_filter();

				return create_alignment_matcher<int16_t>(
					p_query, p_document, metric, metric->alignment_def(), p_matcher_options,
					gen.create_filtered(p_query, p_document, token_filter));
			});

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

		const auto matcher_options = create_alignment_matcher_options(metric->alignment_def());
		if (matcher_options.needs_magnitudes) {
			m_needs_magnitudes = true;
		}

		return MatcherFactory::create(
			matcher_options,
			[p_query, metric, options] (const DocumentRef &p_document, const auto &p_matcher_options) {

				const auto make_tag_weighted_slice = [metric, options] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

					return TagWeightedSlice(
						StaticEmbeddingSlice(metric.get(), slice_id, s, t),
						options);
				};

				const FactoryGenerator gen(make_tag_weighted_slice);
				const auto &token_filter = p_query->token_filter();

				return create_alignment_matcher<int16_t>(
					p_query, p_document, metric, metric->alignment_def(), p_matcher_options,
					gen.create_filtered(p_query, p_document, token_filter));
			});

	} else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}
}

const std::string &StaticEmbeddingMetric::name() const {
	return m_embedding->name();
}
