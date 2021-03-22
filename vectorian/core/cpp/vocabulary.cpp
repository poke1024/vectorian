#include "vocabulary.h"
#include "utils.h"
#include "embedding/static.h"
#include "metric/static.h"
#include "metric/modifier.h"
#include "query.h"

template<typename VocabularyRef>
TokenVectorRef _unpack_tokens(
	VocabularyRef p_vocab,
	const std::shared_ptr<arrow::Table> &p_table,
	const py::list &p_token_strings) {

	const auto idx = numeric_column<arrow::UInt32Type, uint32_t>(p_table, "idx");
	const auto len = numeric_column<arrow::UInt8Type, uint8_t>(p_table, "len");

	/*const auto pos_array = string_column(p_table, "pos");
	const auto tag_array = string_column(p_table, "tag");*/

	const size_t n = idx.size();
	PPK_ASSERT(n == len.size());

	TokenVectorRef tokens_ref = std::make_shared<std::vector<Token>>();

	std::vector<Token> &tokens = *tokens_ref.get();
	std::vector<std::string> token_texts;
	token_texts.reserve(p_token_strings.size());
	for (const auto &s : p_token_strings) {
		token_texts.push_back(s.cast<py::str>());
	}

	/*{
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

    py::object sub = (py::object)py::module::import("re").attr("sub");

    const auto pattern = py::str("[^\\w]");
    const auto repl = py::str("");

    for (size_t i = 0; i < n; i++) {
        auto &t = token_texts[i];
	    try {
	        auto s = py::str(py::str(t).attr("lower")());
			t = py::str(sub(pattern, repl, s));
		} catch (...) {
			std::cerr << "an error occured when processing string: '" << t << "'\n";
			throw;
		}
    }*/

    {
    	py::gil_scoped_release release;

    	//std::lock_guard<std::recursive_mutex> lock(p_vocab->m_mutex);

        for (size_t i = 0; i < n; i++) {
            Token t;
            const std::string &token_text = token_texts[i];

            // std::cout << "token: " << token_text << " " << int(len[i]) << std::endl;
            t.id = p_vocab->add(token_text);

            t.idx = idx[i];
            t.len = len[i];
            t.pos = -1;
            t.tag = -1;
            tokens.push_back(t);
        }

        iterate_strings(p_table, "pos", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).pos = p_vocab->add_pos(s);
        });

        iterate_strings(p_table, "tag", [&tokens, p_vocab] (size_t i, const std::string &s) {
            tokens.at(i).tag = p_vocab->add_tag(s);
        });
	}

	return tokens_ref;
}

TokenVectorRef unpack_tokens(
	const VocabularyRef &p_vocab,
	const std::shared_ptr<arrow::Table> &p_table,
	const py::list &p_token_strings) {
	return _unpack_tokens(p_vocab, p_table, p_token_strings);
}

TokenVectorRef unpack_tokens(
	const QueryVocabularyRef &p_vocab,
	const std::shared_ptr<arrow::Table> &p_table,
	const py::list &p_token_strings) {
	return _unpack_tokens(p_vocab, p_table, p_token_strings);
}

MetricRef QueryVocabulary::create_metric(
	const QueryRef &p_query,
	const py::dict &p_sentence_metric,
	const py::object &p_token_metric) {

	if (p_token_metric.attr("is_modifier").cast<bool>()) {

		std::vector<MetricRef> operand_metrics;

		const auto operands = p_token_metric.attr("operands").cast<py::list>();
		for (const auto &operand : operands) {
			auto metric = create_metric(
				p_query, p_sentence_metric, operand.cast<py::object>());
			operand_metrics.push_back(metric);
		}

		// std::make_shared<ModifiedMetricFactory>

		ModifiedMetricFactory factory(p_token_metric, operand_metrics);
		return factory.create(p_query);

	} else {

		const py::dict token_metric_def =
			p_token_metric.attr("to_args")(p_query->index()).cast<py::dict>();

		const WordMetricDef metric_def{
			token_metric_def["name"].cast<py::str>(),
			token_metric_def["embedding"].cast<py::str>(),
			token_metric_def["metric"].cast<py::object>()};

		const auto embedding_index = m_embedding_manager->to_index(metric_def.embedding);

		const std::vector<EmbeddingRef> embeddings = {
			m_vocab->embedding_manager()->get_compiled(embedding_index),
			embedding_manager()->get_compiled(embedding_index)
		};

		std::vector<StaticEmbeddingRef> static_embeddings;
		static_embeddings.resize(embeddings.size());
		std::transform(embeddings.begin(), embeddings.end(), static_embeddings.begin(), [] (auto x) {
			return std::static_pointer_cast<StaticEmbedding>(x);
		});

		StaticEmbeddingMetricFactory factory(static_embeddings, p_sentence_metric);
		return factory.create(p_query, metric_def);
	}
}