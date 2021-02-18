#ifndef __VECTORIAN_VOCABULARY_H__
#define __VECTORIAN_VOCABULARY_H__

#include "common.h"
#include "embedding/embedding.h"
#include "metric/metric.h"
#include "metric/composite.h"

enum VocabularyMode {
	MODIFY_VOCABULARY,
	DO_NOT_MODIFY_VOCABULARY
};

template<typename T>
class String2Int {
	std::unordered_map<std::string, T> m_strings;

public:
	inline T operator[](const std::string &s) {
		const auto r = m_strings.insert({s, m_strings.size()});
		return r.first->second;
	}

	inline T lookup(const std::string &s) const {
		 const auto r = m_strings.find(s);
		 return r == m_strings.end() ? -1 : r->second;
	}

	std::string inverse_lookup_slow(T t) const {
		for (auto x : m_strings) {
			if (x.second == t) {
				return x.first;
			}
		}
		return "";
	}

	inline size_t size() const {
		return m_strings.size();
	}
};

template<typename T>
class StringLexicon {
	std::unordered_map<std::string, T> m_to_int;
	std::vector<std::string> m_to_str;

public:
	inline T add(const std::string &p_s) {
		const auto r = m_to_int.insert({p_s, m_to_str.size()});
		if (r.second) {
			m_to_str.push_back(p_s);
		}
		return r.first->second;
	}

	inline T lookup(const std::string &p_s) const {
		const auto i = m_to_int.find(p_s);
		if (i != m_to_int.end()) {
			return i->second;
		} else {
			return -1;
		}
	}

	inline const std::string &lookup(T p_id) const {
		return m_to_str.at(p_id);
	}

	inline size_t size() const {
		return m_to_int.size();
	}
};

class Vocabulary {
	String2Int<token_t> m_tokens;

	struct Embedding {
		EmbeddingRef embedding;
		std::vector<token_t> map;
	};

	std::vector<Embedding> m_embeddings;
	std::unordered_map<std::string, size_t> m_embeddings_by_name;
	StringLexicon<int8_t> m_pos;
	StringLexicon<int8_t> m_tag;

	int m_det_pos;

public:
	// basically a mapping from token -> int

	std::recursive_mutex m_mutex;

	Vocabulary() : m_det_pos(-1) {
	}

	int add_embedding(EmbeddingRef p_embedding) {

		std::lock_guard<std::recursive_mutex> lock(m_mutex);
		PPK_ASSERT(m_tokens.size() == 0);

		const size_t next_id = m_embeddings.size();
		m_embeddings_by_name[p_embedding->name()] = next_id;

		Embedding e;
		e.embedding = p_embedding;
		m_embeddings.push_back(e);

		return next_id;
	}

	inline int unsafe_add_pos(const std::string &p_name) {
		const int i = m_pos.add(p_name);
	    if (p_name == "DET") {
	        m_det_pos = i;
	    }
	    return i;
	}

	inline int det_pos() const {
	    return m_det_pos;
	}

	inline int unsafe_add_tag(const std::string &p_name) {
		return m_tag.add(p_name);
	}

	inline token_t unsafe_lookup(const std::string &p_token) const {
		return m_tokens.lookup(p_token);
	}

	inline token_t unsafe_add(const std::string &p_token) {
		const size_t old_size = m_tokens.size();
		const token_t t = m_tokens[p_token];
		if (m_tokens.size() > old_size) { // new token?
			for (Embedding &e : m_embeddings) {
				e.map.push_back(e.embedding->lookup(p_token));
			}
		}
		return t;
	}

	inline std::string token_to_string_slow(token_t p_token) {
		return m_tokens.inverse_lookup_slow(p_token);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos.lookup(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag.lookup(p_tag_id);
	}

	POSWMap mapped_pos_weights(const std::map<std::string, float> &p_pos_weights) const {
		POSWMap pos_weights;
		for (auto const &x : p_pos_weights) {
			const int i = m_tag.lookup(x.first);
			if (i >= 0) {
				pos_weights[i] = x.second;
			}
		}
		return pos_weights;
	}

	MetricRef create_metric(
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		const py::dict &p_metric_def,
		const MetricModifiers &p_modifiers) {

		std::lock_guard<std::recursive_mutex> lock(m_mutex);

		const MetricDef metric_def{
			p_metric_def["name"].cast<py::str>(),
			p_metric_def["embedding"].cast<py::str>(),
			p_metric_def["metric"].cast<py::str>(),
			p_metric_def["options"].cast<py::dict>()};

		if (metric_def.name == "lerp") {

			return std::make_shared<LerpMetric>(
				create_metric(p_needle_text, p_needle, metric_def.options["a"].cast<py::dict>(), p_modifiers),
				create_metric(p_needle_text, p_needle, metric_def.options["b"].cast<py::dict>(), p_modifiers),
				metric_def.options["t"].cast<float>());

		} else if (metric_def.name == "min") {

			return std::make_shared<MinMetric>(
				create_metric(p_needle_text, p_needle, metric_def.options["a"].cast<py::dict>(), p_modifiers),
				create_metric(p_needle_text, p_needle, metric_def.options["b"].cast<py::dict>(), p_modifiers));

		} else if (metric_def.name == "max") {

			return std::make_shared<MaxMetric>(
				create_metric(p_needle_text, p_needle, metric_def.options["a"].cast<py::dict>(), p_modifiers),
				create_metric(p_needle_text, p_needle, metric_def.options["b"].cast<py::dict>(), p_modifiers));

		} else {

			const auto it = m_embeddings_by_name.find(metric_def.embedding);
			if (it == m_embeddings_by_name.end()) {
				std::ostringstream err;
				err << "unknown embedding " << metric_def.embedding << " referenced in metric ";
				throw std::runtime_error(err.str());
			}
			const auto &embedding = m_embeddings[it->second];

			const Eigen::Map<Eigen::Array<token_t, Eigen::Dynamic, 1>> vocabulary_ids(
				const_cast<token_t*>(embedding.map.data()), embedding.map.size());

			return embedding.embedding->create_metric(
				metric_def,
				vocabulary_ids,
				p_needle_text,
				p_needle,
				p_modifiers);
		}
	}
};

typedef std::shared_ptr<Vocabulary> VocabularyRef;

TokenVectorRef unpack_tokens(
	VocabularyRef p_vocab,
	VocabularyMode p_mode,
	const std::string p_text,
	const std::shared_ptr<arrow::Table> &p_table);

#endif // __VECTORIAN_VOCABULARY_H__
