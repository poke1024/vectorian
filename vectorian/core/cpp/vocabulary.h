#ifndef __VECTORIAN_VOCABULARY_H__
#define __VECTORIAN_VOCABULARY_H__

#include "common.h"
#include "embedding/embedding.h"
#include "metric/metric.h"
#include "metric/composite.h"
#include <iostream>

/*template<typename T>
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
};*/

template<typename T>
class LexiconBase {
public:
	inline constexpr T to_id(const std::string &p_s) const {
		return -1;
	}

	inline const std::string &to_str(T p_id) const {
		std::ostringstream err;
		err << "illegal call to LexiconBase::to_str() with id " << p_id;
		throw std::runtime_error(err.str());
	}

	inline constexpr size_t size() const {
		return 0;
	}
};

template<typename T>
using LexiconBaseRef = std::shared_ptr<LexiconBase<T>>;

template<typename T, typename BaseRef>
class LexiconImpl {
	const BaseRef m_base;
	const std::string m_unknown;
	std::unordered_map<std::string, T> m_to_int;
	std::vector<std::string> m_to_str;

public:
	LexiconImpl(const BaseRef p_base) : m_base(p_base), m_unknown("<unk>") {
		//std::cout << "creating lexicon " << this << " base is at " << &m_base << "\n";
		//std::cout << "base has " << m_base->size() << " items\n";
	}

	virtual ~LexiconImpl() {
		//std::cout << "destroying lexicon " << this << "\n";
	}

	inline T add(const std::string &p_s) {
		const T i = m_base->to_id(p_s);
		if (i >= 0) {
			return i;
		}
		const auto r = m_to_int.insert({
			p_s, m_to_str.size() + m_base->size()});
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
			return m_base->to_id(p_s);
		}
	}

	inline const std::string &to_str(T p_id) const {
		if (p_id < 0) {
			return m_unknown;
		} if (static_cast<size_t>(p_id) < m_base->size()) {
			return m_base->to_str(p_id);
		} else {
			return m_to_str.at(p_id - m_base->size());
		}
	}

	inline void reserve(const size_t p_size) {
		m_to_str.reserve(std::max(
			ssize_t(0),
			ssize_t(p_size) - ssize_t(m_base->size())));
	}

	inline size_t size() const {
		return m_base->size() + m_to_int.size();
	}

	inline size_t inc_offset() const {
		return m_base->size();
	}

	inline const std::vector<std::string>& inc_strings() const {
		return m_to_str;
	}
};

template<typename T>
class Lexicon;

template<typename T>
using LexiconRef = std::shared_ptr<Lexicon<T>>;

template<typename T>
class IncrementalLexicon : public LexiconImpl<T, LexiconRef<T>> {
public:
	IncrementalLexicon(const LexiconRef<T> &p_lexicon) :
		LexiconImpl<T, LexiconRef<T>>(p_lexicon) {
	}
};

template<typename T>
using IncrementalLexiconRef = std::shared_ptr<IncrementalLexicon<T>>;

template<typename T>
class Lexicon : public
	LexiconImpl<T, LexiconBaseRef<T>>,
	std::enable_shared_from_this<Lexicon<T>> {
public:
	Lexicon() : LexiconImpl<T, LexiconBaseRef<T>>(
		std::make_shared<LexiconBase<T>>()) {
	}
};

template<typename T>
IncrementalLexiconRef<T> make_incremental(const LexiconRef<T> &p_lexicon) {
	return std::make_shared<IncrementalLexicon<T>>(p_lexicon);
}


class QueryVocabulary;

class Vocabulary {
protected:
	friend class QueryVocabulary;

	const LexiconRef<token_t> m_tokens;
	const LexiconRef<int8_t> m_pos;
	const LexiconRef<int8_t> m_tag;

	struct Embedding {
		EmbeddingRef embedding;
		std::vector<token_t> map;
	};

	struct EmbeddingTokenRef {
		int16_t embedding;
		token_t token;
	};

	std::unordered_map<std::string, size_t> m_embeddings_by_name;
	std::vector<Embedding> m_embeddings;

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

	Vocabulary() :
		m_tokens(std::make_shared<Lexicon<token_t>>()),
		m_pos(std::make_shared<Lexicon<int8_t>>()),
		m_tag(std::make_shared<Lexicon<int8_t>>()) {

		//std::cout << "creating vocabulary at " << this << "\n";
	}

	virtual ~Vocabulary() {
		//std::cout << "destroying vocabulary at " << this << "\n";
	}

	inline size_t size() const {
		return m_tokens->size();
	}

	int add_embedding(EmbeddingRef p_embedding) {

		std::lock_guard<std::recursive_mutex> lock(m_mutex);

		if (m_tokens->size() > 0) {
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

	inline int add_pos(const std::string &p_name) {
		return m_pos->add(p_name);
	}

	inline int unsafe_pos_id(const std::string &p_name) const {
		return m_pos->to_id(p_name);
	}

	inline int add_tag(const std::string &p_name) {
		return m_tag->add(p_name);
	}

	inline int unsafe_tag_id(const std::string &p_name) const {
		return m_tag->to_id(p_name);
	}

	inline token_t token_to_id(const std::string &p_token) const {
		return m_tokens->to_id(p_token);
	}

	inline token_t add(const std::string &p_token) {
		return m_tokens->add(p_token);
	}

	const std::vector<token_t> &get_embedding_map(const int p_emb_idx) {
		auto &e = m_embeddings[p_emb_idx];
		e.embedding->update_map(
			e.map, m_tokens->inc_strings(), m_tokens->inc_offset());
		return e.map;
	}

	inline const std::string &id_to_token(token_t p_token) const {
		return m_tokens->to_str(p_token);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos->to_str(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag->to_str(p_tag_id);
	}
};

typedef std::shared_ptr<Vocabulary> VocabularyRef;

class QueryVocabulary {
	const VocabularyRef m_vocab;

	const IncrementalLexiconRef<token_t> m_tokens;
	const IncrementalLexiconRef<int8_t> m_pos;
	const IncrementalLexiconRef<int8_t> m_tag;

	struct Embedding {
		EmbeddingRef embedding;
		std::vector<token_t> extra_map;
	};

	std::vector<Embedding> m_embeddings;

public:
	QueryVocabulary(const VocabularyRef &p_vocab) :
		m_vocab(p_vocab),
		m_tokens(make_incremental(m_vocab->m_tokens)),
		m_pos(make_incremental(m_vocab->m_pos)),
		m_tag(make_incremental(m_vocab->m_tag)) {

		for (const auto &e : m_vocab->m_embeddings) {
			m_embeddings.push_back(Embedding{e.embedding});
		}

		//std::cout << "creating query vocabulary at " << this << "\n";
	}

	virtual ~QueryVocabulary() {
		//std::cout << "destroying query vocabulary at " << this << "\n";
	}

	inline size_t size() const {
		return m_tokens->size();
	}

	inline const std::string &id_to_token(const token_t p_token) const {
		return m_tokens->to_str(p_token);
	}

	inline token_t add(const std::string &p_token) {
		return m_tokens->add(p_token);
	}

	const std::vector<token_t> &get_embedding_map(const int p_emb_idx) {
		auto &e = m_embeddings[p_emb_idx];
		e.embedding->update_map(
			e.extra_map, m_tokens->inc_strings(), m_tokens->inc_offset());
		return e.extra_map;
	}

	inline int add_pos(const std::string &p_name) {
		return m_pos->add(p_name);
	}

	inline int add_tag(const std::string &p_name) {
		return m_tag->add(p_name);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos->to_str(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag->to_str(p_tag_id);
	}

	MetricRef create_metric(
		const std::vector<Token> &p_needle,
		const py::dict &p_sent_metric_def,
		const py::dict &p_word_metric_def) {

		const WordMetricDef metric_def{
			p_word_metric_def["name"].cast<py::str>(),
			p_word_metric_def["embedding"].cast<py::str>(),
			p_word_metric_def["metric"].cast<py::str>(),
			p_word_metric_def["options"].cast<py::dict>()};

		if (metric_def.name == "lerp") {

			return std::make_shared<LerpMetric>(
				create_metric(p_needle, p_sent_metric_def, metric_def.options["a"].cast<py::dict>()),
				create_metric(p_needle, p_sent_metric_def, metric_def.options["b"].cast<py::dict>()),
				metric_def.options["t"].cast<float>());

		} else if (metric_def.name == "min") {

			return std::make_shared<MinMetric>(
				create_metric(p_needle, p_sent_metric_def, metric_def.options["a"].cast<py::dict>()),
				create_metric(p_needle, p_sent_metric_def, metric_def.options["b"].cast<py::dict>()));

		} else if (metric_def.name == "max") {

			return std::make_shared<MaxMetric>(
				create_metric(p_needle, p_sent_metric_def, metric_def.options["a"].cast<py::dict>()),
				create_metric(p_needle, p_sent_metric_def, metric_def.options["b"].cast<py::dict>()));

		} else {

			const auto it = m_vocab->m_embeddings_by_name.find(metric_def.embedding);
			if (it == m_vocab->m_embeddings_by_name.end()) {
				std::ostringstream err;
				err << "unknown embedding " << metric_def.embedding << " referenced in metric " << metric_def.name;
				throw std::runtime_error(err.str());
			}

			const auto &map0 = m_vocab->get_embedding_map(it->second);
			const auto &map1 = get_embedding_map(it->second);
			const std::vector<MappedTokenIdArray> vocabulary_ids = {
				MappedTokenIdArray(const_cast<token_t*>(map0.data()), map0.size()),
				MappedTokenIdArray(const_cast<token_t*>(map1.data()), map1.size())
			};

			return m_embeddings[it->second].embedding->create_metric(
				metric_def,
				p_sent_metric_def,
				vocabulary_ids,
				p_needle);
		}
	}

	POSWMap mapped_pos_weights(
		const std::map<std::string, float> &p_pos_weights) const {

		POSWMap pos_weights;
		for (auto const &x : p_pos_weights) {
			const int i = m_tag->to_id(x.first);
			if (i >= 0) {
				pos_weights[i] = x.second;
			} else {
				std::ostringstream err;
				err << "unknown Penn Treebank tag " << x.first;
				throw std::runtime_error(err.str());
			}
		}
		return pos_weights;
	}
};

typedef std::shared_ptr<QueryVocabulary> QueryVocabularyRef;

TokenVectorRef unpack_tokens(
	const VocabularyRef &p_vocab,
	const std::shared_ptr<arrow::Table> &p_table,
	const py::list &p_token_strings);

TokenVectorRef unpack_tokens(
	const QueryVocabularyRef &p_vocab,
	const std::shared_ptr<arrow::Table> &p_table,
	const py::list &p_token_strings);

#endif // __VECTORIAN_VOCABULARY_H__
