#include "vocabulary.h"
#include "utils.h"


TokenVectorRef unpack_tokens(
	VocabularyRef p_vocab,
	VocabularyMode p_mode,
	const std::string p_text,
	const std::shared_ptr<arrow::Table> &p_table) {

	const auto idx = numeric_column<arrow::UInt32Type, uint32_t>(p_table, "idx");
	const auto len = numeric_column<arrow::UInt8Type, uint8_t>(p_table, "len");

	/*const auto pos_array = string_column(p_table, "pos");
	const auto tag_array = string_column(p_table, "tag");*/

	const size_t n = idx.size();
	PPK_ASSERT(n == len.size());

	TokenVectorRef tokens_ref = std::make_shared<std::vector<Token>>();

	std::vector<Token> &tokens = *tokens_ref.get();
	std::vector<std::string> token_texts;

	{
    	py::gil_scoped_release release;

    	tokens.reserve(n);
    	token_texts.reserve(n);

        for (size_t i = 0; i < n; i++) {
            if (idx[i] + len[i] > p_text.length()) {
                std::ostringstream s;
                s << "illegal token[" << i << "].idx + .len = " <<
                    size_t(idx[i]) << " + " << size_t(len[i]) << " > " << p_text.length();
                throw std::runtime_error(s.str());
            }

            token_texts.push_back(p_text.substr(idx[i], len[i]));
        }
    }

    py::object sub = (py::object) py::module::import("re").attr("sub");

    auto pattern = py::str("[^\\w]");
    auto repl = py::str("");

    for (size_t i = 0; i < n; i++) {
        auto &t = token_texts[i];
        auto s = py::str(py::str(t).attr("lower")());
		t = py::str(sub(pattern, repl, s));
    }

    {
    	py::gil_scoped_release release;

    	std::lock_guard<std::mutex> lock(p_vocab->m_mutex);

        for (size_t i = 0; i < n; i++) {
            Token t;
            const std::string &token_text = token_texts[i];

            // std::cout << "token: " << token_text << " " << int(len[i]) << std::endl;
            t.id = (p_mode == MODIFY_VOCABULARY) ?
                p_vocab->unsafe_add(token_text) :
                p_vocab->unsafe_lookup(token_text);

            t.idx = idx[i];
            t.len = len[i];
            t.pos = -1;
            t.tag = -1;
            tokens.push_back(t);
        }

        iterate_strings(p_table, "pos", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).pos = p_vocab->unsafe_add_pos(s);
        });

        iterate_strings(p_table, "tag", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).tag = p_vocab->unsafe_add_tag(s);
        });
	}

	return tokens_ref;
}
