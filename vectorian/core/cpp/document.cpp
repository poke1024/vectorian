#include "document.h"
#include "query.h"
#include "match/match.h"

std::vector<VariableSpans::Span> unpack_spans(const py::dict &p_table) {
	typedef VariableSpans::offset_t offset_t;

	const auto start_array = p_table["start"].cast<py::array_t<offset_t>>();
	const auto end_array = p_table["end"].cast<py::array_t<offset_t>>();

	const ssize_t n = start_array.shape(0);
	PPK_ASSERT(end_array.shape(0) == n);

	std::vector<VariableSpans::Span> spans;
	spans.reserve(n);

	const auto start_read = start_array.unchecked<1>();
	const auto end_read = end_array.unchecked<1>();

	for (ssize_t i = 0; i < n; i++) {
		spans.emplace_back(VariableSpans::Span{
			start_read(i),
			end_read(i)
		});
	}

	return spans;
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

ResultSetRef Document::find(
    const QueryRef &p_query,
    const BoosterRef &p_booster) {

	if (m_tokens->empty()) {
		return ResultSetRef();
	}

	return p_query->match(shared_from_this(), p_booster);
}
