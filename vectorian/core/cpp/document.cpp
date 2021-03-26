#include "document.h"
#include "query.h"

std::vector<int32_t> unpack_spans(const py::dict &p_table) {
	const auto n_tokens_array = p_table["n_tokens"].cast<py::array_t<uint16_t>>();
	const auto n_tokens_read = n_tokens_array.unchecked<1>();

	const size_t n = n_tokens_array.shape(0);
	std::vector<int32_t> offsets;
	offsets.reserve(n + 1);

	size_t token_at = 0;
	offsets.push_back(token_at);
	for (size_t i = 0; i < n; i++) {
		token_at += n_tokens_read(i);
		offsets.push_back(token_at);
	}

	return offsets;
}

Document::Document(
	const int64_t p_document_id,
	VocabularyRef p_vocab,
	const py::dict &p_spans,
	const py::dict &p_tokens,
	const py::dict &p_metadata,
	const py::dict &p_contextual_embeddings):

	m_id(p_document_id),
	m_vocab(p_vocab),
	m_metadata(p_metadata) {

	for (auto item : p_contextual_embeddings) {
		m_contextual_vectors[item.first.cast<py::str>()] = item.second.cast<py::object>();
	}

	for (auto item : p_spans) {
		m_spans[item.first.cast<py::str>()] = std::make_shared<Spans>(
			VariableSpans(unpack_spans(item.second.cast<py::dict>())));
	}

	m_tokens = unpack_tokens(p_vocab, p_tokens);

	m_spans["token"] = std::make_shared<Spans>(
		FixedSpans(m_tokens->size()));

	m_num_dummy_tokens = 1;
	add_dummy_token(*m_tokens.get());
}

ResultSetRef Document::find(const QueryRef &p_query) {

	if (m_tokens->empty()) {
		return ResultSetRef();
	}

	return p_query->match(shared_from_this());
}
