#include "common.h"
#include "query.h"
#include "match/matcher.h"
#include "result_set.h"
#include "metric/static.h"

ResultSetRef Query::match(
	const DocumentRef &p_document) {

	ResultSetRef matches = std::make_shared<ResultSet>(
		max_matches(), min_score());

	for (const auto &metric : m_metrics) {
		const auto matcher = metric->matcher_factory()->create_matcher(
			shared_from_this(), metric, p_document);

		matcher->initialize();

		{
			py::gil_scoped_release release;
			matcher->match(matches);
		}
	}

	return matches;
}

void Query::initialize(
	const py::object &p_tokens_table,
	const py::list &p_tokens_strings,
	py::kwargs p_kwargs) {

	// FIXME: assert that we are in main thread here.

	const std::shared_ptr<arrow::Table> table(
	    unwrap_table(p_tokens_table.ptr()));

	m_t_tokens = unpack_tokens(
		m_vocab, table, p_tokens_strings);

	m_vocab->compile_embeddings();

	m_py_t_tokens = to_py_array(m_t_tokens);

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

	m_token_filter.pos = parse_filter_mask(p_kwargs, "pos_filter",
		[this] (const std::string &s) -> int {
			return m_vocab->base()->unsafe_pos_id(s);
		});
	m_token_filter.tag = parse_filter_mask(p_kwargs, "tag_filter",
		[this] (const std::string &s) -> int {
			return m_vocab->base()->unsafe_tag_id(s);
		});

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

		/*const py::dict token_metric_def =
			p_token_metric.attr("to_args")(p_query->index()).cast<py::dict>();

		const WordMetricDef metric_def{
			token_metric_def["name"].cast<py::str>(),
			token_metric_def["embedding"].cast<py::str>(),
			token_metric_def["metric"].cast<py::object>()};*/

		// const auto it = m_contextual_embeddings.find(metric_def.embedding);
		// embedding->create_metric(shared_from_this(),
		//	const QueryRef &p_query,
		//	const WordMetricDef &p_word_metric,
		//	const py::dict &p_sent_metric_def,

		StaticEmbeddingMatcherFactoryFactory matcher_ff(metric_def_dict);
		const auto matcher_factory = matcher_ff.create_matcher_factory(shared_from_this());

		auto strategy = m_vocab->create_strategy(
			shared_from_this(),
			matcher_factory,
			metric_def_dict["token_metric"].cast<py::object>());

		auto m = std::make_shared<StaticEmbeddingMetric>(
			strategy.name,
			strategy.matrix_factory->create(DocumentRef()),
			matcher_factory
		);

		m_metrics.push_back(m);
	}
}
