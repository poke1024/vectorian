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

MatcherFactoryRef StaticEmbeddingMatcherFactoryFactory::create_matcher_factory(
	const QueryRef &p_query) {

	py::gil_scoped_acquire acquire;

	const std::string sentence_metric_kind =
		m_sent_metric_def["metric"].cast<py::str>();

	const py::dict alignment_def = this->alignment_def();

	if (sentence_metric_kind == "alignment-isolated") {

		const auto matcher_options = create_alignment_matcher_options(alignment_def);

		return MatcherFactory::create(
			matcher_options,
			[alignment_def] (

				const QueryRef &p_query,
				const MetricRef &p_metric,
				const DocumentRef &p_document,
				const auto &p_matcher_options) {

				const auto matrix = std::static_pointer_cast<StaticEmbeddingMetric>(p_metric)->matrix();

				const SliceFactoryFactory gen_slices([matrix] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

			        return StaticEmbeddingSlice<int16_t>(*matrix.get(), slice_id, s, t);
				});

				return create_alignment_matcher<int16_t>(
					p_query, p_document, p_metric, alignment_def, p_matcher_options,
					gen_slices.create_filtered(p_query, p_document, p_query->token_filter()));
			});

	} else if (sentence_metric_kind == "alignment-tag-weighted") {

		TagWeightedOptions options;
		options.t_pos_weights = parse_tag_weights(p_query, m_sent_metric_def["tag_weights"]);
		options.pos_mismatch_penalty = m_sent_metric_def["pos_mismatch_penalty"].cast<float>();
		options.similarity_threshold = m_sent_metric_def["similarity_threshold"].cast<float>();

		float sum = 0.0f;
		for (float w : options.t_pos_weights) {
			sum += w;
		}
		options.t_pos_weights_sum = sum;

		const auto matcher_options = create_alignment_matcher_options(alignment_def);

		return MatcherFactory::create(
			matcher_options,
			[alignment_def, options] (

				const QueryRef &p_query,
				const MetricRef &p_metric,
				const DocumentRef &p_document,
				const auto &p_matcher_options) {

				const auto matrix = std::static_pointer_cast<StaticEmbeddingMetric>(p_metric)->matrix();

				const SliceFactoryFactory gen_slices([matrix, options] (
					const size_t slice_id,
					const TokenSpan &s,
					const TokenSpan &t) {

					return TagWeightedSlice(
						StaticEmbeddingSlice<int16_t>(*matrix.get(), slice_id, s, t),
						options);
				});

				return create_alignment_matcher<int16_t>(
					p_query, p_document, p_metric, alignment_def, p_matcher_options,
					gen_slices.create_filtered(p_query, p_document, p_query->token_filter()));
			});

	} else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}
}

// --------------------------------------------------------------------------------

SimilarityMatrixRef StaticEmbeddingSimilarityBuilder::build_similarity_matrix(
	const QueryRef &p_query,
	const WordMetricDef &p_metric) {

	const QueryVocabularyRef p_vocabulary = p_query->vocabulary();
	const Needle needle(p_query);

	const auto matrix = std::make_shared<SimilarityMatrix>();

	const size_t vocab_size = p_vocabulary->size();
	const size_t needle_size = static_cast<size_t>(needle.size());
	matrix->m_similarity.resize({ssize_t(vocab_size), ssize_t(needle_size)});

	const auto &needle_tokens = needle.token_ids();
	py::list sources;
	py::array_t<size_t> indices{static_cast<py::ssize_t>(needle_size)};
	auto mutable_indices = indices.mutable_unchecked<1>();

	for (size_t j = 0; j < needle_size; j++) { // for each token in needle
		const auto t = needle_tokens[j];
		size_t t_rel;
		const auto &t_vectors = pick_vectors(m_embeddings, t, t_rel);
		sources.append(t_vectors);
		mutable_indices[j] = t_rel;
	}

	const auto py_embeddings = py::module_::import("vectorian.embeddings");
	const auto needle_vectors = py_embeddings.attr("StackedVectors")(sources, indices);

	size_t offset = 0;
	for (const auto &embedding : m_embeddings) {
		const auto &vectors = embedding->vectors();
		const size_t size = embedding->size();

		p_metric.vector_metric(
			vectors,
			needle_vectors,
			xt::strided_view(matrix->m_similarity, {xt::range(offset, offset + size), xt::all()}));

		PPK_ASSERT(offset + size <= vocab_size);

		offset += size;
	}
	PPK_ASSERT(offset == vocab_size);

	for (size_t j = 0; j < needle.size(); j++) { // for each token in needle

		// since the j-th needle token is a specific vocabulary token, we always
		// set that specific vocabulary token similarity to 1 (regardless of the
		// embedding distance).
		const auto k = needle_tokens[j];
		if (k >= 0) {
			matrix->m_similarity(k, j) = 1.0f;
		}
	}

	return matrix;
}

SimilarityMatrixRef StaticEmbeddingSimilarityBuilder::create(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const MatcherFactoryRef &p_matcher_factory) {

	const auto matrix = build_similarity_matrix(
		p_query,
		p_metric);

	//std::cout << "has debug hook " << p_query->debug_hook().has_value() << "\n";
	/*if (p_query->debug_hook().has_value()) {
		auto gen_rows = py::cpp_function([&] () {
			const auto &vocab = p_query->vocabulary();

			py::list row_tokens;
			const size_t n = vocab->size();
			for (size_t i = 0; i < n; i++) {
				row_tokens.append(vocab->id_to_token(i));
			}
			return row_tokens;
		});

		auto gen_columns = py::cpp_function([&] () {
			const auto &vocab = p_query->vocabulary();

			py::list col_tokens;
			for (const auto &t : *p_query->tokens()) {
				col_tokens.append(vocab->id_to_token(t.id));
			}
			return col_tokens;
		});

		py::dict data;
		data["matrix"] = m_similarity;
		data["rows"] = gen_rows;
		data["columns"] = gen_columns;

		(*p_query->debug_hook())("similarity_matrix", data);
	}*/

	if (p_matcher_factory->needs_magnitudes()) {
		const Needle needle(p_query);

		compute_magnitudes(
			matrix,
			p_query->vocabulary(),
			needle);
	}

	return matrix;

	//return std::make_shared<StaticEmbeddingMetric>(
	//	m_embeddings[0]->name(), matrix, matcher_factory);
}
