#ifndef __VECTORIAN_VOCABULARY_H__
#define __VECTORIAN_VOCABULARY_H__

#include "common.h"
#include "embedding/embedding.h"
#include "metric/metric.h"
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

class EmbeddingManager;
typedef std::shared_ptr<EmbeddingManager> EmbeddingManagerRef;

class EmbeddingManager {
	struct Embedding {
		py::object embedding; // py embedding session instance
		py::str py_name;
		bool is_static;
		py::object to_core;
		EmbeddingRef compiled;
	};

	std::unordered_map<std::string, size_t> m_embeddings_by_name;
	std::vector<Embedding> m_embeddings;
	bool m_is_compiled;

public:
	EmbeddingManager() : m_is_compiled(false) {
	}

	EmbeddingManager(const EmbeddingManager &p_other) :
		m_embeddings_by_name(p_other.m_embeddings_by_name),
		m_embeddings(p_other.m_embeddings),
		m_is_compiled(false) {

		for (auto &e : m_embeddings) {
			e.compiled.reset();
		}
	}

	EmbeddingManagerRef clone() const {
		return std::make_shared<EmbeddingManager>(*this);
	}

	size_t to_index(const std::string &p_name) const {
		const auto it = m_embeddings_by_name.find(p_name);
		if (it == m_embeddings_by_name.end()) {
			std::ostringstream err;
			err << "unknown embedding \"" << p_name << "\". did you miss to add it to your session?";
			throw std::runtime_error(err.str());
		}
		return it->second;
	}

	inline bool is_static(const size_t p_index) const {
		return m_embeddings.at(p_index).is_static;
	}

	inline py::str get_py_name(const size_t p_index) const {
		return m_embeddings.at(p_index).py_name;
	}

	size_t add_embedding(py::object p_embedding) {

		if (m_is_compiled) {
			throw std::runtime_error("EmbeddingManager cannot add new embeddings after compilation");
		}

		const size_t next_id = m_embeddings.size();
		m_embeddings_by_name[p_embedding.attr("name").cast<std::string>()] = next_id;

		Embedding e;
		e.embedding = p_embedding;
		e.py_name = p_embedding.attr("name").cast<py::str>();
		e.is_static = p_embedding.attr("is_static").cast<bool>();
		e.to_core = p_embedding.attr("to_core");
		m_embeddings.push_back(e);

		return next_id;
	}

	void compile_static(const py::list &p_tokens) {
		for (auto &e : m_embeddings) {
			if (e.is_static) {
				e.compiled = e.to_core(p_tokens).template cast<EmbeddingRef>();
			}
		}
		m_is_compiled = true;
	}

	void compile_contextual() {
		for (auto &e : m_embeddings) {
			if (!e.is_static) {
				e.compiled = e.to_core().template cast<EmbeddingRef>();
			}
		}
		m_is_compiled = true;
	}

	const EmbeddingRef &get_compiled(const size_t p_index) const {
		return m_embeddings.at(p_index).compiled;
	}
};


class Vocabulary {
protected:
	friend class QueryVocabulary;

	const EmbeddingManagerRef m_embedding_manager;

	const LexiconRef<token_t> m_tokens;
	const LexiconRef<int8_t> m_pos;
	const LexiconRef<int8_t> m_tag;

	struct EmbeddingTokenRef {
		int16_t embedding;
		token_t token;
	};

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

	Vocabulary(const EmbeddingManagerRef &p_embedding_manager) :
		m_embedding_manager(p_embedding_manager),
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

	void compile_embeddings() {
		py::list tokens;
		for (const auto &s : m_tokens->inc_strings()) {
			tokens.append(py::str(s));
		}
		if (tokens.size() == 0) {
			throw std::runtime_error("no tokens in vocabulary");
		}

		m_embedding_manager->compile_static(tokens);
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

	/*const std::vector<token_t> &get_embedding_map(const int p_emb_idx) {
		auto &e = m_embeddings[p_emb_idx];
		e.embedding->update_map(
			e.map, m_tokens->inc_strings(), m_tokens->inc_offset());
		return e.map;
	}*/

	inline const std::string &id_to_token(token_t p_token) const {
		return m_tokens->to_str(p_token);
	}

	inline const std::string &pos_str(int8_t p_pos_id) {
		return m_pos->to_str(p_pos_id);
	}

	inline const std::string &tag_str(int8_t p_tag_id) {
		return m_tag->to_str(p_tag_id);
	}

	inline const EmbeddingManagerRef &embedding_manager() const {
		return m_embedding_manager;
	}
};

typedef std::shared_ptr<Vocabulary> VocabularyRef;


class QueryVocabulary {
	const VocabularyRef m_vocab;
	const EmbeddingManagerRef m_embedding_manager;

	const IncrementalLexiconRef<token_t> m_tokens;
	const IncrementalLexiconRef<int8_t> m_pos;
	const IncrementalLexiconRef<int8_t> m_tag;

public:
	QueryVocabulary(const VocabularyRef &p_vocab) :
		m_vocab(p_vocab),
		m_embedding_manager(p_vocab->embedding_manager()->clone()),
		m_tokens(make_incremental(m_vocab->m_tokens)),
		m_pos(make_incremental(m_vocab->m_pos)),
		m_tag(make_incremental(m_vocab->m_tag)) {

		//std::cout << "creating query vocabulary at " << this << "\n";
	}

	virtual ~QueryVocabulary() {
		//std::cout << "destroying query vocabulary at " << this << "\n";
	}

	inline const VocabularyRef &base() const {
		return m_vocab;
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

	void compile_embeddings() {
		py::list tokens;
		for (const auto &s : m_tokens->inc_strings()) {
			tokens.append(py::str(s));
		}

		m_embedding_manager->compile_static(tokens);
	}

	/*const std::vector<token_t> &get_embedding_map(const int p_emb_idx) {
		auto &e = m_embeddings[p_emb_idx];
		e.embedding->update_map(
			e.extra_map, m_tokens->inc_strings(), m_tokens->inc_offset());
		return e.extra_map;
	}*/

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

	inline const EmbeddingManagerRef &embedding_manager() const {
		return m_embedding_manager;
	}

	std::vector<StaticEmbeddingRef> get_compiled_embeddings(const size_t p_embedding_index) const;
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
