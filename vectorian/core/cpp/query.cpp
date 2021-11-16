#include "common.h"
#include "query.h"
#include "match/matcher.h"
#include "result_set.h"
#include "metric/static.h"
#include "metric/contextual.h"
#include "metric/modifier.h"
#include "match/instantiate.h"

ResultSetRef Query::match(
	const DocumentRef &p_document,
	const BoosterRef &p_booster) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), Score(min_score(), 1));

	for (const auto &metric : m_metrics) {
		const auto matcher = metric->matcher_factory()->create_matcher(
			shared_from_this(), metric, p_document, p_booster);

		matcher->initialize();

		{
			py::gil_scoped_release release;
			matcher->match(matches);
		}
	}

	return matches;
}

void Query::initialize(
	const py::dict &p_tokens,
	py::kwargs p_kwargs) {

	// FIXME: assert that we are in main thread here.

	m_t_tokens = unpack_tokens(m_vocab, p_tokens);

	m_vocab->compile_embeddings();

	m_py_t_tokens = to_py_array(
		m_t_tokens, m_t_tokens->size());

	static const std::set<std::string> valid_options = {
		"metric",
		"pos_filter",
		"tag_filter",
		"submatch_weight",
		"bidirectional",
		"max_matches",
		"min_score",
		"partition",
		"debug"
	};

	if (p_kwargs) {
		for (auto item : p_kwargs) {
			const std::string name = py::str(item.first);
			if (valid_options.find(name) == valid_options.end()) {
				std::ostringstream err;
				err << "illegal query option " << name;
				throw std::runtime_error(err.str());
			}
#if 0
			const std::string value = py::str(item.second);
			std::cout << "received param " << name << ": " <<
				value << "\n";
#endif
		}
	}

	if (p_kwargs && p_kwargs.contains("debug")) {
		m_debug_hook = p_kwargs["debug"].cast<py::object>();
	}

	m_submatch_weight = (p_kwargs && p_kwargs.contains("submatch_weight")) ?
        p_kwargs["submatch_weight"].cast<float>() :
        0.0f;

	m_bidirectional = (p_kwargs && p_kwargs.contains("bidirectional")) ?
        p_kwargs["bidirectional"].cast<bool>() :
        false;

    m_token_filter = make_token_filter(p_kwargs);

	m_max_matches = (p_kwargs && p_kwargs.contains("max_matches")) ?
		p_kwargs["max_matches"].cast<size_t>() :
		100;

	m_min_score = (p_kwargs && p_kwargs.contains("min_score")) ?
		p_kwargs["min_score"].cast<float>() :
		0.2f;

	if (p_kwargs && p_kwargs.contains("partition")) {
		const auto slices_def_dict = p_kwargs["partition"].cast<py::dict>();

		m_slice_strategy.level = slices_def_dict["level"].cast<py::str>();
		m_slice_strategy.window_size = slices_def_dict["window_size"].cast<size_t>();
		m_slice_strategy.window_step = slices_def_dict["window_step"].cast<size_t>();

		if (m_slice_strategy.window_size < 1) {
			throw std::runtime_error("partition window size needs to be >= 1");
		}
		if (m_slice_strategy.window_step < 1) {
			throw std::runtime_error("partition window step needs to be >= 1");
		}

	} else {
		m_slice_strategy.level = "sentence";
		m_slice_strategy.window_size = 1;
		m_slice_strategy.window_step = 1;
	}

	if (p_kwargs && p_kwargs.contains("metric")) {
		const auto metric_def_dict = p_kwargs["metric"].cast<py::dict>();

		const auto matcher_factory = create_matcher_factory(
			shared_from_this(),
			metric_def_dict);

		const auto strategy = create_strategy(
			matcher_factory,
			metric_def_dict["token_metric"].cast<py::object>());

		MetricRef metric;

		switch (strategy.type) {
			case STATIC: {
				const auto matrix = strategy.matrix_factory->create(
					STATIC, DocumentRef());
				if (debug_hook().has_value()) {
					matrix->call_hook(shared_from_this());
				}
				metric = std::make_shared<StaticEmbeddingMetric>(
					strategy.name,
					matrix,
					matcher_factory);
			} break;
			case CONTEXTUAL: {
				metric = std::make_shared<ContextualEmbeddingMetric>(
					strategy.name,
					strategy.matrix_factory,
					matcher_factory
				);
			} break;
			default: {
				throw std::runtime_error("unsupported embedding type");
			}
		}

		m_metrics.push_back(metric);
	}
}

Query::Strategy Query::create_strategy(
	const MatcherFactoryRef &p_matcher_factory,
	const py::object &p_token_metric) {

	if (p_token_metric.attr("is_modifier").cast<bool>()) {

		std::vector<SimilarityMatrixFactoryRef> operands;
		EmbeddingType type = STATIC;

		for (const auto &operand : p_token_metric.attr("operands").cast<py::list>()) {
			auto strategy = create_strategy(
				p_matcher_factory, operand.cast<py::object>());
			if (strategy.type == CONTEXTUAL) {
				type = CONTEXTUAL;
			}
			operands.push_back(strategy.matrix_factory);
		}

		return Strategy{
			type,
			p_token_metric.attr("name").cast<py::str>(),
			std::make_shared<ModifiedSimilarityMatrixFactory>(
				p_token_metric, operands)
		};

	} else {

		const py::dict token_metric_def =
			p_token_metric.attr("to_args")(index()).cast<py::dict>();

		const WordMetricDef metric_def{
			token_metric_def["name"].cast<py::str>(),
			token_metric_def["embedding"].cast<py::str>(),
			token_metric_def["metric"].cast<py::object>()};

		const auto embedding_manager = m_vocab->embedding_manager();
		const auto embedding_index = embedding_manager->to_index(metric_def.embedding);

		switch (embedding_manager->get_embedding_type(embedding_index)) {
			case STATIC: {
				const auto sim_factory = std::make_shared<StaticEmbeddingSimilarityMatrixFactory>(
					shared_from_this(), metric_def, p_matcher_factory, embedding_index);

				return Strategy{
					STATIC,
					metric_def.name,
					sim_factory};
			}
			case CONTEXTUAL: {
				const auto sim_factory = std::make_shared<ContextualEmbeddingSimilarityMatrixFactory>(
					shared_from_this(), metric_def, p_matcher_factory, embedding_index);

				return Strategy{
					CONTEXTUAL,
					metric_def.name,
					sim_factory};
			}
			default: {
				throw std::runtime_error("unsupported embedding type");
			}
		}
	}
}

TokenFilterRef Query::make_token_filter(const py::kwargs &p_kwargs) const {
	const auto pos_mask = parse_filter_mask(p_kwargs, "pos_filter",
		[this] (const std::string &s) -> int {
			return m_vocab->base()->unsafe_pos_id(s);
		});
	const auto tag_mask = parse_filter_mask(p_kwargs, "tag_filter",
		[this] (const std::string &s) -> int {
			return m_vocab->base()->unsafe_tag_id(s);
		});

	py::list tokens;
	if (p_kwargs && p_kwargs.contains("token_filter")) {
		tokens = p_kwargs["token_filter"].cast<py::list>();
	}

	if (pos_mask == 0 && tag_mask == 0 && tokens.size() == 0) {
		return TokenFilterRef();
	} else {
		auto tf = std::make_shared<TokenFilter>(pos_mask, tag_mask);

		if (tokens.size() > 0) {

			tf->vocab = xt::xtensor<bool, 1>();
			auto &mask = *(tf->vocab);
			mask.resize({m_vocab->size()});
			mask.fill(true);

			for (const auto &t : tokens) {
				const auto k = m_vocab->token_to_id(t.cast<py::str>());
				if (k >= 0) {
					mask(k) = false;
				}
			}
		}

		return tf;
	}
}


SliceStrategy::SliceStrategy(const py::dict &p_slice_strategy) {
	level = p_slice_strategy["level"].cast<py::str>();
	window_size = p_slice_strategy["window_size"].cast<py::int_>();
	window_step = p_slice_strategy["window_step"].cast<py::int_>();
}
