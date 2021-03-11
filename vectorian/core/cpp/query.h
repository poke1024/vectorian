#ifndef __VECTORIAN_QUERY_H__
#define __VECTORIAN_QUERY_H__

#include "common.h"
#include "utils.h"
#include "metric/composite.h"
#include "vocabulary.h"

struct TokenFilter {
	uint64_t pos;
	uint64_t tag;

	inline bool all() const {
		return !(pos || tag);
	}

	inline bool operator()(const Token &t) const {
		return !(((pos >> t.pos) & 1) || ((tag >> t.tag) & 1));
	}
};

template<typename Lookup>
uint64_t parse_filter_mask(
	const py::kwargs &p_kwargs,
	const char *p_filter_name,
	Lookup lookup) {

	uint64_t filter = 0;
	if (p_kwargs && p_kwargs.contains(p_filter_name)) {
		for (auto x : p_kwargs[p_filter_name].cast<py::list>()) {
			const std::string s = x.cast<py::str>();
			const int i = lookup(s);
			if (i < 0) {
				std::ostringstream err;
				err << "illegal value " << s << " for " << p_filter_name;
				throw std::runtime_error(err.str());
			}
			filter |= static_cast<uint64_t>(1) << i;
		}
	}
	return filter;
}

struct SliceStrategy {
	std::string level; // e.g. "sentence"
	size_t window_size;
	size_t window_step;
};

class Query : public std::enable_shared_from_this<Query> {

	struct MatchStrategy {
		MetricRef metric;
		MatcherFactoryRef matcher_factory;
	};

	const QueryVocabularyRef m_vocab;
	std::vector<MatchStrategy> m_match_strategies;
	TokenVectorRef m_t_tokens;
	py::dict m_py_t_tokens;
	float m_submatch_weight;
	bool m_bidirectional;
	TokenFilter m_token_filter;
	bool m_aborted;
	size_t m_max_matches;
	float m_min_score;
	POSWMap m_pos_weights;
	std::vector<float> m_t_tokens_pos_weights;
	SliceStrategy m_slice_strategy;
	std::optional<py::object> m_debug_hook;

public:
	Query(VocabularyRef p_vocab) :

		m_vocab(std::make_shared<QueryVocabulary>(p_vocab)),
		m_aborted(false) {
	}

	void initialize(
		const py::object &p_tokens_table,
		const py::list &p_tokens_strings,
		py::kwargs p_kwargs) {

		// FIXME: assert that we are in main thread here.

		const std::shared_ptr<arrow::Table> table(
		    unwrap_table(p_tokens_table.ptr()));

		m_t_tokens = unpack_tokens(
			m_vocab, table, p_tokens_strings);

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

			const auto metric = m_vocab->create_metric(
				shared_from_this(),
				metric_def_dict,
				metric_def_dict["token_metric"]);

			const auto matcher_factory =
				metric->create_matcher_factory(shared_from_this());

			m_match_strategies.emplace_back(MatchStrategy{metric, matcher_factory});
		}
	}

	virtual ~Query() {
		//std::cout << "destroying query at " << this << "\n";
	}

	const QueryVocabularyRef &vocabulary() const {
		return m_vocab;
	}

	inline const TokenVectorRef &tokens() const {
		return m_t_tokens;
	}

	inline py::dict py_tokens() const {
		return to_py_array(m_t_tokens);
	}

	inline int len() const {
		return m_t_tokens->size();
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	const std::vector<MatchStrategy> &match_strategies() const {
		return m_match_strategies;
	}

	inline bool bidirectional() const {
		return m_bidirectional;
	}

	inline const TokenFilter &token_filter() const {
	    return m_token_filter;
	}

	ResultSetRef match(
		const DocumentRef &p_document);

	bool aborted() const {
		return m_aborted;
	}

	void abort() {
		m_aborted = true;
	}

	inline size_t max_matches() const {
		return m_max_matches;
	}

	inline float min_score() const {
		return m_min_score;
	}

	inline float submatch_weight() const {
		return m_submatch_weight;
	}

	inline const SliceStrategy& slice_strategy() const {
		return m_slice_strategy;
	}

	inline const std::optional<py::object>& debug_hook() const {
		return m_debug_hook;
	}

	template<typename Slice>
	py::dict make_py_debug_slice(const Slice &p_slice) const {
		const QueryVocabularyRef vocab = this->vocabulary();

		const auto token_vector = [&] (const auto &get_id, const int n) {
			py::list id;
			py::list text;
			for (int i = 0; i < n; i++) {
				id.append(get_id(i));
				text.append(vocab->id_to_token(get_id(i)));
			}
			py::dict tokens;
			tokens["id"] = id;
			tokens["text"] = text;
			return tokens;
		};

		py::dict data;

		data["s"] = token_vector([&] (int i) {
			return p_slice.s(i).id;
		}, p_slice.len_s());

		data["t"] = token_vector([&] (int i) {
			return p_slice.t(i).id;
		}, p_slice.len_t());

		return data;
	}
};

typedef std::shared_ptr<Query> QueryRef;

#endif // __VECTORIAN_QUERY_H__
