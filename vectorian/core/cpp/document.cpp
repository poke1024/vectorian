#include "document.h"
#include "utils.h"
#include "query.h"

std::vector<int32_t> unpack_spans(const std::shared_ptr<arrow::Table> &p_table) {
	const auto n_tokens_values = numeric_column<arrow::UInt16Type, uint16_t>(p_table, "n_tokens");

	const size_t n = n_tokens_values.size();
	std::vector<int32_t> offsets;
	offsets.reserve(n + 1);

	size_t token_at = 0;
	offsets.push_back(token_at);
	for (size_t i = 0; i < n; i++) {
		token_at += n_tokens_values[i];
		offsets.push_back(token_at);
	}

	return offsets;
}

Document::Document(
	const int64_t p_document_id,
	VocabularyRef p_vocab,
	const py::dict &p_spans,
	const py::object &p_tokens_table,
	const py::list &p_tokens_strings,
	const py::dict &p_metadata):

	m_id(p_document_id),
	m_vocab(p_vocab),
	m_metadata(p_metadata) {

	for (auto item : p_spans) {
		const auto table = unwrap_table(item.second.cast<py::object>());
		m_spans[item.first.cast<py::str>()] = std::make_shared<Spans>(unpack_spans(table));
	}

	const auto tokens_table = unwrap_table(p_tokens_table);
	m_tokens = unpack_tokens(
		p_vocab, tokens_table, p_tokens_strings);

	add_dummy_token(*m_tokens.get());

	{
		py::gil_scoped_acquire acquire;
		m_py_tokens = to_py_array(m_tokens);
	}
}

ResultSetRef Document::find(const QueryRef &p_query) {

	if (m_tokens->empty()) {
		return ResultSetRef();
	}

	return p_query->match(shared_from_this());
}
