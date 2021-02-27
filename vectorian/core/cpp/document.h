#ifndef __VECTORIAN_DOCUMENT_H__
#define __VECTORIAN_DOCUMENT_H__

#include "common.h"
#include "vocabulary.h"

inline void add_dummy_token(std::vector<Token> &tokens) {
	if (tokens.empty()) {
		return;
	}
	// adding a last dummy token with the correct idx is handy.
	Token t;
	t.id = -1;
	t.idx = tokens.rbegin()->idx + tokens.rbegin()->len;
	t.len = 0;
	t.pos = -1;
	t.tag = -1;
	tokens.push_back(t);
}

class Document : public std::enable_shared_from_this<Document> {
private:
	const int64_t m_id;
	const VocabularyRef m_vocab;

	TokenVectorRef m_tokens;
	py::dict m_py_tokens;
	std::vector<Sentence> m_sentences;
	size_t m_max_len_s;

	const py::dict m_metadata;
	std::string m_cache_path;

public:
	Document(
		int64_t p_document_id,
		VocabularyRef p_vocab,
		const py::object &p_sentences,
		const py::object &p_tokens_table,
		const py::list &p_tokens_strings,
		const py::dict &p_metadata,
		const std::string p_cache_path);

	ResultSetRef find(const QueryRef &p_query);

	inline VocabularyRef vocabulary() const {
		return m_vocab;
	}

	std::string __str__() const {
		return "<vectorian.core.Document " +
			m_metadata["author"].cast<std::string>() +
			", " +
			m_metadata["title"].cast<std::string>() + ">";
	}

	inline int64_t id() const {
		return m_id;
	}

	const std::string &path() const {
		return m_cache_path;
	}

	const py::dict &metadata() const {
		return m_metadata;
	}

	inline py::dict py_tokens() const {
		return m_py_tokens;
	}

	inline const TokenVectorRef &tokens() const {
		return m_tokens;
	}

	inline size_t n_tokens() const {
		return m_tokens->size();
	}

	inline const std::vector<Sentence> &sentences() const {
		return m_sentences;
	}

	size_t n_sentences() const {
		return m_sentences.size();
	}

	inline size_t max_len_s() const { // maximum sentence length (in tokens)
		return m_max_len_s;
	}

	inline const Sentence &sentence(size_t p_index) const {
		return m_sentences.at(p_index);
	}
};

typedef std::shared_ptr<Document> DocumentRef;

#endif // __VECTORIAN_DOCUMENT_H__
