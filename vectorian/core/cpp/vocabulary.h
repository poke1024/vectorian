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

	inline T to_id(const std::string &p_s) const {
		const auto i = m_to_int.find(p_s);
		if (i != m_to_int.end()) {
			return i->second;
		} else {
			return -1;
		}
	}

	inline const std::string &to_str(T p_id) const {
		return m_to_str.at(p_id);
	}

	inline void reserve(const size_t p_size) {
		m_to_str.reserve(p_size);
	}

	inline size_t size() const {
		return m_to_int.size();
	}
};

class Vocabulary {
	StringLexicon<token_t> m_tokens;

	struct Embedding {
		EmbeddingRef embedding;
		std::vector<token_t> map;
	};

	struct EmbeddingTokenRef {
		int16_t embedding;
		token_t token;
	};

	std::vector<Embedding> m_embeddings;
	std::unordered_map<std::string, size_t> m_embeddings_by_name;
	StringLexicon<int8_t> m_pos;
	StringLexicon<int8_t> m_tag;

	/*void init_with_embedding_tokens() {
		PPK_ASSERT(m_tokens.size() == 0);

		std::vector<const std::vector<std::string>*> emb_tokens;
		for (Embedding &e : m_embeddings) {
			emb_tokens.push_back(&e.embedding->tokens());
		}

		if (m_embeddings.size() == 1) {
			const auto &tokens = *emb_tokens[0];

			m_tokens.reserve(tokens.size());
			for (Embedding &e : m_embeddings) {
				e.map.reserve(tokens.size());
			}

			auto &m = m_embeddings[0].map;
			size_t index = 0;
			for (const auto &s : tokens) {
				m_tokens.add(s);
				m.push_back(index);
				index += 1;
			}
		} else {
			size_t total_token_ref_count = 0;
			for (size_t emb_index = 0; emb_index < m_embeddings.size(); emb_index++) {
				total_token_ref_count += emb_tokens[emb_index]->size();
			}

			std::vector<EmbeddingTokenRef> emb_token_refs;
			emb_token_refs.reserve(total_token_ref_count);

			for (size_t emb_index = 0; emb_index < m_embeddings.size(); emb_index++) {
				const size_t n_tokens = emb_tokens[emb_index]->size();
				for (size_t tok_index = 0; tok_index < n_tokens; tok_index++) {
					emb_token_refs.emplace_back(EmbeddingTokenRef{
						static_cast<int16_t>(emb_index),
						static_cast<token_t>(tok_index)
					});
				}
			}

			std::sort(emb_token_refs.begin(), emb_token_refs.end(),
				[&emb_tokens] (const auto &a, const auto &b) {
					return (*emb_tokens[a.embedding])[a.token] <
						(*emb_tokens[b.embedding])[b.token];
				});

			const std::string *last_token = nullptr;
			for (const auto &t : emb_token_refs) {
				const std::string &s = (*emb_tokens[t.embedding])[t.token];

				if (!last_token || s != *last_token) {
					m_tokens.add(s);
					last_token = &s;

					for (Embedding &e : m_embeddings) {
						e.map.push_back(-1);
					}
				}

				m_embeddings[t.embedding].map.back() = t.token;
			}
		}
	}*/

public:
	// basically a mapping from token -> int

	std::recursive_mutex m_mutex;

	Vocabulary() {
	}

	inline size_t size() const {
		return m_tokens.size();
	}

	int add_embedding(EmbeddingRef p_embedding) {

		std::lock_guard<std::recursive_mutex> lock(m_mutex);

		if (m_tokens.size() > 0) {
			throw std::runtime_error(
				"cannot add embeddings after tokens were added.");
		}

		const size_t next_id = m_embeddings.size();
		m_embeddings_by_name[p_embedding->name()] = next_id;

		Embedding e;
		e.embedding = p_embedding;
		m_embeddings.push_back(e);

		return next_id;
	}

	inline int unsafe_add_pos(const std::string &p_name) {
		const int i = m_pos.add(p_name);
	    return i;
	}

	inline int unsafe_pos_id(const std::string &p_name) const {
		return m_pos.to_id(p_name);
	}

	inline int unsafe_add_tag(const std::string &p_name) {
		return m_tag.add(p_name);
	}

	inline int unsafe_tag_id(const std::string &p_name) const {
		return m_tag.to_id(p_name);
	}

	inline token_t unsafe_lookup(const std::string &p_token) const {
		return m_tokens.to_id(p_token);
	}

	inline token_t unsafe_add(const std::string &p_token) {
		const size_t old_size = m_tokens.size();
		/*if (old_size == 0) {
			init_with_embedding_tokens();
		}*/

		const token_t t = m_tokens.add(p_token);
		if (m_tokens.size() > old_size) { // new token?
			for (Embedding &e : m_embeddings) {
				e.map.push_back(e.embedding->lookup(p_token));
			}
		}

		return t;
	}

	inline const std::string &id_to_token(token_t p_token) const {
		return m_tokens.to_str(p_token);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos.to_str(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag.to_str(p_tag_id);
	}

	POSWMap mapped_pos_weights(const std::map<std::string, float> &p_pos_weights) const {
		POSWMap pos_weights;
		for (auto const &x : p_pos_weights) {
			const int i = m_tag.to_id(x.first);
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
