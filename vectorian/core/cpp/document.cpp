#include "document.h"
#include "utils.h"
#include "query.h"

std::vector<Sentence> unpack_sentences(const std::shared_ptr<arrow::Table> &p_table) {
	py::gil_scoped_release release;

	const auto book = numeric_column<arrow::Int8Type, int8_t>(p_table, "book");
	const auto chapter = numeric_column<arrow::Int8Type, int8_t>(p_table, "chapter");
	const auto speaker = numeric_column<arrow::Int8Type, int8_t>(p_table, "speaker");
	const auto location = numeric_column<arrow::UInt16Type, uint16_t>(p_table, "location");

	const auto n_tokens_values = numeric_column<arrow::UInt16Type, uint16_t>(p_table, "n_tokens");

	const size_t n = n_tokens_values.size();
	std::vector<Sentence> sentences;
	sentences.reserve(n);

	int32_t token_at = 0;
	for (size_t i = 0; i < n; i++) {
		 Sentence s;
		 s.book = book[i];
		 s.chapter = chapter[i];
		 s.speaker = speaker[i];
		 s.paragraph = location[i];
		 s.n_tokens = n_tokens_values[i];

		 s.token_at = token_at;
		 token_at += s.n_tokens;

		 sentences.push_back(s);
	}

	return sentences;
}

Document::Document(
	int64_t p_document_id,
	VocabularyRef p_vocab,
	const std::string &p_text,
	const py::object &p_sentences,
	const py::object &p_tokens,
	const py::dict &p_metadata,
	const std::string &p_cache_path = std::string()):

	m_id(p_document_id),
	m_vocab(p_vocab),
	m_text(p_text),
	m_metadata(p_metadata),
	m_cache_path(p_cache_path) {

	const auto sentences_table = unwrap_table(p_sentences);
	m_sentences = unpack_sentences(sentences_table);

	const auto tokens_table = unwrap_table(p_tokens);
	m_tokens = unpack_tokens(
		p_vocab, MODIFY_VOCABULARY, m_text, tokens_table);

	add_dummy_token(*m_tokens.get());

	{
		py::gil_scoped_acquire acquire;
		m_py_tokens = to_py_array(m_tokens);
	}

	size_t max_len = 0;
	for (const auto &s : m_sentences) {
		max_len = std::max(max_len, size_t(s.n_tokens));
	}
	m_max_len_s = max_len;
}

ResultSetRef Document::find(const QueryRef &p_query) {

	if (m_tokens->empty()) {
		return ResultSetRef();
	}

	return p_query->match(shared_from_this());
}
