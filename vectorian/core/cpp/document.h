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

class Spans {
	const std::vector<int32_t> m_offsets;
	size_t m_max_len;

public:
	Spans(std::vector<int32_t> &&p_offsets) : m_offsets(p_offsets) {
		size_t max_len = 0;
		if (m_offsets.size() > 0) {
			for (size_t i = 0; i < m_offsets.size() - 1; i++) {
				max_len = std::max(
					max_len,
					size_t(m_offsets[i + 1] - m_offsets[i]));
			}
		}
		m_max_len = max_len;
	}

	inline size_t size() const {
		// we model spans and have one more offset than spans.
		return m_offsets.size() - 1;
	}

	inline int32_t start(const size_t p_index) const {
		return m_offsets[p_index];
	}

	inline int32_t end(const size_t p_index) const {
		return m_offsets[p_index + 1];
	}

	inline int32_t len(const size_t p_index) const {
		return m_offsets[p_index + 1] - m_offsets[p_index];
	}

	inline int32_t safe_len(const size_t p_index, const size_t p_size) const {
		const size_t i1 = std::min(p_index + p_size, m_offsets.size() - 1);
		return m_offsets[i1] - m_offsets[p_index];
	}

	inline Slice slice(const size_t p_index) const {
		const auto i0 = m_offsets.at(p_index);
		const auto i1 = m_offsets.at(p_index + 1);
		return Slice{i0, i1 - i0};
	}

	inline int32_t max_len() const {
		return m_max_len;
	}
};

typedef std::shared_ptr<Spans> SpansRef;


class Document : public std::enable_shared_from_this<Document> {
private:
	const int64_t m_id;
	const VocabularyRef m_vocab;

	TokenVectorRef m_tokens;
	py::dict m_py_tokens;
	std::map<std::string, SpansRef> m_spans;

	const py::dict m_metadata;
	std::string m_cache_path;

public:
	Document(
		int64_t p_document_id,
		VocabularyRef p_vocab,
		const py::dict &p_spans,
		const py::object &p_tokens_table,
		const py::list &p_tokens_strings,
		const py::dict &p_metadata);

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

	const SpansRef &spans(const std::string &p_name) const {
		const auto it = m_spans.find(p_name);
		if (it == m_spans.end()) {
			std::ostringstream err;
			err << "unknown spans " << p_name;
			throw std::runtime_error(err.str());
		}
		return it->second;
	}

	inline size_t max_len(const std::string &p_name) const {
		return spans(p_name)->max_len();
	}

	/*inline const std::vector<int32_t> &offsets() const {
		return m_offsets; // into sentences
	}

	size_t n_sentences() const {
		return m_offsets.size() - 1;
	}

	inline size_t max_len_s() const { // maximum sentence length (in tokens)
		return m_max_len_s;
	}

	inline Slice sentence(size_t p_index) const {
		const auto i0 = m_offsets.at(p_index);
		const auto i1 = m_offsets.at(p_index + 1);
		return Slice{i0, i1 - i0};
	}*/
};

typedef std::shared_ptr<Document> DocumentRef;

#endif // __VECTORIAN_DOCUMENT_H__
