#ifndef __VECTORIAN_QUERY_H__
#define __VECTORIAN_QUERY_H__

#include "common.h"
#include "utils.h"
#include "metric/composite.h"
#include "vocabulary.h"

class Query : public std::enable_shared_from_this<Query> {
	py::dict m_alignment_algorithm;
	std::vector<MetricRef> m_metrics;
	const std::string m_text;
	TokenVectorRef m_t_tokens;
	POSWMap m_pos_weights;
	std::vector<float> m_t_tokens_pos_weights;
	float m_total_score;
	int m_mismatch_length_penalty;
	float m_submatch_weight;
	bool m_bidirectional;
	bool m_ignore_determiners;
	bool m_aborted;
	size_t m_max_matches;
	float m_min_score;

public:
	Query(
		VocabularyRef p_vocab,
		const std::string &p_text,
		py::handle p_tokens_table,
		py::kwargs p_kwargs) : m_text(p_text), m_aborted(false) {

		const std::shared_ptr<arrow::Table> table(
		    unwrap_table(p_tokens_table.ptr()));

		m_t_tokens = unpack_tokens(
			p_vocab, DO_NOT_MODIFY_VOCABULARY, p_text, table);

		static const std::set<std::string> valid_options = {
			"alignment",
			"metrics",
			"pos_mismatch_penalty",
			"pos_weights",
			"similarity_falloff",
			"similarity_threshold",
			"submatch_weight",
			"bidirectional",
			"ignore_determiners",
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

		m_alignment_algorithm = (p_kwargs && p_kwargs.contains("alignment")) ?
            p_kwargs["alignment"].cast<py::dict>() : py::dict();

		const float pos_mismatch_penalty =
			(p_kwargs && p_kwargs.contains("pos_mismatch_penalty")) ?
				p_kwargs["pos_mismatch_penalty"].cast<float>() :
				1.0f;

		const float similarity_threshold = (p_kwargs && p_kwargs.contains("similarity_threshold")) ?
            p_kwargs["similarity_threshold"].cast<float>() :
            0.0f;

		const float similarity_falloff = (p_kwargs && p_kwargs.contains("similarity_falloff")) ?
            p_kwargs["similarity_falloff"].cast<float>() :
            1.0f;

		m_submatch_weight = (p_kwargs && p_kwargs.contains("submatch_weight")) ?
            p_kwargs["submatch_weight"].cast<float>() :
            0.0f;

		m_bidirectional = (p_kwargs && p_kwargs.contains("bidirectional")) ?
            p_kwargs["bidirectional"].cast<bool>() :
            false;

		m_ignore_determiners = (p_kwargs && p_kwargs.contains("ignore_determiners")) ?
            p_kwargs["ignore_determiners"].cast<bool>() :
            false;

		m_max_matches = (p_kwargs && p_kwargs.contains("max_matches")) ?
			p_kwargs["max_matches"].cast<size_t>() :
			100;

		m_min_score = (p_kwargs && p_kwargs.contains("min_score")) ?
			p_kwargs["min_score"].cast<float>() :
			0.2f;

		std::map<std::string, float> pos_weights;
		if (p_kwargs && p_kwargs.contains("pos_weights")) {
			auto pws = p_kwargs["pos_weights"].cast<py::dict>();
			for (const auto &pw : pws) {
				pos_weights[pw.first.cast<py::str>()] = pw.second.cast<py::float_>();
			}
		}

		m_pos_weights = p_vocab->mapped_pos_weights(pos_weights);

		m_t_tokens_pos_weights.reserve(m_t_tokens->size());
		for (size_t i = 0; i < m_t_tokens->size(); i++) {
			const Token &t = m_t_tokens->at(i);

			auto w = m_pos_weights.find(t.tag);
			float s;
			if (w != m_pos_weights.end()) {
				s = w->second;
			} else {
				s = 1.0f;
			}

			m_t_tokens_pos_weights.push_back(s);
		}

		m_total_score = 0.0f;
		for (float w : m_t_tokens_pos_weights) {
			m_total_score += w;
		}

		if (p_kwargs && p_kwargs.contains("metrics")) {
			MetricModifiers modifiers;
			modifiers.pos_mismatch_penalty = pos_mismatch_penalty;
			modifiers.similarity_falloff = similarity_falloff;
			modifiers.similarity_threshold = similarity_threshold;
			modifiers.pos_weights = m_pos_weights;

			const auto given_metric_defs = p_kwargs["metrics"].cast<py::list>();
			for (auto metric_def : given_metric_defs) {
				m_metrics.push_back(p_vocab->create_metric(
					m_text,
					*m_t_tokens.get(),
					metric_def.cast<py::dict>(),
					modifiers));
			}
		}
	}

	const std::string &text() const {
		return m_text;
	}

	inline const TokenVectorRef &tokens() const {
		return m_t_tokens;
	}

	inline int len() const {
		return m_t_tokens->size();
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	const py::dict &alignment_algorithm() const {
		return m_alignment_algorithm;
	}

	const std::vector<MetricRef> &metrics() const {
		return m_metrics;
	}

	inline int mismatch_length_penalty() const {
		return m_mismatch_length_penalty;
	}

	inline bool bidirectional() const {
		return m_bidirectional;
	}

	inline bool ignore_determiners() const {
	    return m_ignore_determiners;
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

	inline float reference_score(
		const float p_matched,
		const float p_unmatched) const {

		// m_matched_weight == 0 indicates that there
		// is no higher relevance of matched content than
		// unmatched content, both are weighted equal (see
		// maximum_internal_score()).

		const float unmatched_weight = std::pow(
			(m_total_score - p_matched) / m_total_score,
			m_submatch_weight);

		const float reference_score =
			p_matched +
			unmatched_weight * (m_total_score - p_matched);

		return reference_score;
	}

	inline float normalized_score(
		const float p_raw_score,
		const std::vector<int16_t> &p_match) const {

		// unboosted version would be:
		// return p_raw_score / m_total_score;

		// a final boosting step allowing matched content
		// more weight than unmatched content.

		const size_t n = p_match.size();

		float matched_score = 0.0f;
		float unmatched_score = 0.0f;

		for (size_t i = 0; i < n; i++) {

			const float s = m_t_tokens_pos_weights[i];

			if (p_match[i] < 0) {
				unmatched_score += s;
			} else {
				matched_score += s;
			}
		}

		return p_raw_score / reference_score(matched_score, unmatched_score);
	}
};

typedef std::shared_ptr<Query> QueryRef;

#endif // __VECTORIAN_QUERY_H__
