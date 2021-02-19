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
	const std::string m_text;

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
		const std::string &p_text,
		const py::object &p_sentences,
		const py::object &p_tokens,
		const py::dict &p_metadata,
		const std::string &p_cache_path);

	ResultSetRef find(const QueryRef &p_query);

	inline VocabularyRef vocabulary() const {
		return m_vocab;
	}

	std::string __str__() const {
		return "<cpp.vcore.Document " +
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

	const std::string &text() const {
		return m_text;
	}

	std::string substr(ssize_t p_start, ssize_t p_end) const {
		return m_text.substr(p_start, std::max(ssize_t(0), p_end - p_start));
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

	py::dict py_sentence(size_t p_index) const {
		const Sentence &s = m_sentences.at(p_index);
		py::dict d;
		d["book"] = s.book;
		d["chapter"] = s.chapter;
		d["speaker"] = s.speaker;
		d["paragraph"] = s.paragraph;
		d["token_at"] = s.token_at;
		d["n_tokens"] = s.n_tokens;
		return d;
	}

	py::list py_sentences_as_tokens() const {
		size_t k = 0;
		py::list py_doc;
		const auto &tokens = *m_tokens.get();
		for (const Sentence &s : m_sentences) {
			py::list py_sent;
			for (int i = 0; i < s.n_tokens; i++) {
				const auto &t = tokens[k++];
				py_sent.append(py::str(m_text.substr(t.idx, t.len)));
			}
			py_doc.append(py_sent);
		}
		return py_doc;
	}

	py::list py_sentences_as_text() const {
		py::list py_sentences;
		const auto &tokens = *m_tokens.get();
		for (const Sentence &s : m_sentences) {
			if (s.n_tokens > 0) {
				const auto &t0 = tokens[s.token_at];

				int32_t i1;
				if (s.token_at + s.n_tokens < static_cast<int32_t>(tokens.size())) {
					i1 = tokens[s.token_at + s.n_tokens].idx;
				} else {
					const auto &t1 = tokens[s.token_at + s.n_tokens - 1];
					i1 = t1.idx + t1.len;
				}

				py_sentences.append(py::str(m_text.substr(t0.idx, i1 - t0.idx)));
			} else {
				py_sentences.append(py::str(""));
			}
		}
		return py_sentences;
	}

};

typedef std::shared_ptr<Document> DocumentRef;

#endif // __VECTORIAN_DOCUMENT_H__
