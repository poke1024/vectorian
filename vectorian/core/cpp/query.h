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
	const QueryVocabularyRef m_vocab;
	std::vector<MetricRef> m_metrics;
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

public:
	Query(
		VocabularyRef p_vocab,
		const py::object &p_tokens_table,
		const py::list &p_tokens_strings,
		py::kwargs p_kwargs) :

		m_vocab(std::make_shared<QueryVocabulary>(p_vocab)),
		m_aborted(false) {

		// FIXME: assert that we are in main thread here.

		const std::shared_ptr<arrow::Table> table(
		    unwrap_table(p_tokens_table.ptr()));

		m_t_tokens = unpack_tokens(
			p_vocab, table, p_tokens_strings);

		m_py_t_tokens = to_py_array(m_t_tokens);

		static const std::set<std::string> valid_options = {
			"metric",
			"pos_filter",
			"tag_filter",
			"submatch_weight",
			"bidirectional",
			"max_matches",
			"min_score"
		};

		if (p_kwargs) {
			for (auto item : p_kwargs) {
				const std::string name = py::str(item.first);
				if (valid_options.find(name) == valid_options.end()) {
					std::ostringstream err;
					err << "illegal option " << name;
					throw std::runtime_error(err.str());
				}
#if 0
				const std::string value = py::str(item.second);
				std::cout << "received param " << name << ": " <<
					value << "\n";
#endif
			}
		}

		m_submatch_weight = (p_kwargs && p_kwargs.contains("submatch_weight")) ?
            p_kwargs["submatch_weight"].cast<float>() :
            0.0f;

		m_bidirectional = (p_kwargs && p_kwargs.contains("bidirectional")) ?
            p_kwargs["bidirectional"].cast<bool>() :
            false;

		m_token_filter.pos = parse_filter_mask(p_kwargs, "pos_filter",
			[&p_vocab] (const std::string &s) -> int {
				return p_vocab->unsafe_pos_id(s);
			});
		m_token_filter.tag = parse_filter_mask(p_kwargs, "tag_filter",
			[&p_vocab] (const std::string &s) -> int {
				return p_vocab->unsafe_tag_id(s);
			});

		m_max_matches = (p_kwargs && p_kwargs.contains("max_matches")) ?
			p_kwargs["max_matches"].cast<size_t>() :
			100;

		m_min_score = (p_kwargs && p_kwargs.contains("min_score")) ?
			p_kwargs["min_score"].cast<float>() :
			0.2f;

		if (p_kwargs && p_kwargs.contains("metric")) {
			const auto metric_def_dict = p_kwargs["metric"].cast<py::dict>();

			m_metrics.push_back(m_vocab->create_metric(
				*m_t_tokens.get(),
				metric_def_dict,
				metric_def_dict["word_metric"]));
		}

		if (p_kwargs && p_kwargs.contains("slices")) {
			const auto slices_def_dict = p_kwargs["slices"].cast<py::dict>();

			m_slice_strategy.level = slices_def_dict["level"].cast<py::str>();
			m_slice_strategy.window_size = slices_def_dict["window_size"].cast<size_t>();
			m_slice_strategy.window_step = slices_def_dict["window_step"].cast<size_t>();

		} else {
			m_slice_strategy.level = "sentence";
			m_slice_strategy.window_size = 1;
			m_slice_strategy.window_step = 1;
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

	const std::vector<MetricRef> &metrics() const {
		return m_metrics;
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
};

typedef std::shared_ptr<Query> QueryRef;

#endif // __VECTORIAN_QUERY_H__
