#ifndef __VECTORIAN_DOCUMENT_H__
#define __VECTORIAN_DOCUMENT_H__

#include "common.h"
#include "vocabulary.h"
#include "embedding/vectors.h"

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

class FixedSpans {
	const size_t m_size;

public:
	FixedSpans(const size_t p_size) : m_size(p_size) {
	}

	inline size_t size() const {
		return m_size;
	}

	inline int32_t start(const size_t p_index) const {
		return p_index;
	}

	inline int32_t end(const size_t p_index) const {
		return p_index + 1;
	}

	inline int32_t len(const size_t p_index) const {
		return 1;
	}

	inline int32_t bounded_len(const size_t p_index, const size_t p_size) const {
		return std::min(p_size, m_size - p_index);
	}

	inline int32_t max_len(const size_t p_window_size) const {
		return p_window_size;
	}
};


class VariableSpans {
	const std::vector<int32_t> m_offsets;
	size_t m_max_len;

public:
	VariableSpans(std::vector<int32_t> &&p_offsets) : m_offsets(p_offsets) {
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
		return end(p_index) - start(p_index);
	}

	inline int32_t bounded_len(const size_t p_index, const size_t p_size) const {
		const size_t i1 = std::min(p_index + p_size, m_offsets.size() - 1);
		return start(i1) - start(p_index);
	}

	inline int32_t max_len(const size_t p_window_size) const {
		return m_max_len * p_window_size;
	}
};

class Spans {
	std::optional<FixedSpans> m_fixed;
	std::optional<VariableSpans> m_variable;

public:
	Spans(FixedSpans &&p_spans) : m_fixed(p_spans) {
	}

	Spans(VariableSpans &&p_spans) : m_variable(p_spans) {
	}

	inline size_t size() const {
		return m_variable.has_value() ? (*m_variable).size() : (*m_fixed).size();
	}

	inline int32_t start(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).start(p_index) : (*m_fixed).start(p_index);
	}

	inline int32_t end(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).end(p_index) : (*m_fixed).end(p_index);
	}

	inline int32_t len(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).len(p_index) : (*m_fixed).len(p_index);
	}

	inline int32_t bounded_len(const size_t p_index, const size_t p_size) const {
		return m_variable.has_value() ?
			(*m_variable).bounded_len(p_index, p_size) :
			(*m_fixed).bounded_len(p_index, p_size);
	}

	inline Slice slice(const size_t p_index) const {
		return Slice{start(p_index), len(p_index)};
	}

	inline int32_t max_len(const size_t p_window_size) const {
		return m_variable.has_value() ? (*m_variable).max_len(p_window_size) : (*m_fixed).max_len(p_window_size);
	}
};

typedef std::shared_ptr<Spans> SpansRef;


class Document :
	public std::enable_shared_from_this<Document>,
	public ContextualVectorsContainer {

private:
	const int64_t m_id;
	const VocabularyRef m_vocab;

	TokenVectorRef m_tokens;
	size_t m_num_dummy_tokens;
	std::map<std::string, SpansRef> m_spans;

	const py::dict m_metadata;
	std::string m_cache_path;

public:
	Document(
		int64_t p_document_id,
		VocabularyRef p_vocab,
		const py::dict &p_spans,
		const py::dict &p_tokens,
		const py::dict &p_metadata,
		const py::dict &p_contextual_embeddings);

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

	py::dict py_tokens() const {
		return to_py_array(m_tokens, n_tokens());
	}

	inline const TokenVectorRef &tokens() const {
		return m_tokens;
	}

	inline size_t n_tokens() const {
		return m_tokens->size() - m_num_dummy_tokens;
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

	inline size_t max_len(const std::string &p_name, const size_t p_window_size) const {
		return spans(p_name)->max_len(p_window_size);
	}
};

typedef std::shared_ptr<Document> DocumentRef;

#endif // __VECTORIAN_DOCUMENT_H__
