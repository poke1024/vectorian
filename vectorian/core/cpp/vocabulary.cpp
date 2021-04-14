#include "vocabulary.h"
#include "embedding/static.h"
#include "metric/static.h"
#include "metric/modifier.h"
#include "query.h"
#include "document.h"

template<typename VocabularyRef>
TokenVectorRef _unpack_tokens(
	const VocabularyRef &p_vocab,
	const py::dict &p_tokens) {

	const auto idx_array = p_tokens["idx"].cast<py::array_t<uint32_t>>();
	const auto len_array = p_tokens["len"].cast<py::array_t<uint8_t>>();
	const auto str_array = p_tokens["str"].cast<py::list>();
	const auto pos_array = p_tokens["pos"].cast<py::list>();
	const auto tag_array = p_tokens["tag"].cast<py::list>();

	const ssize_t n = idx_array.shape(0);
	PPK_ASSERT(n == len_array.shape(0));
	PPK_ASSERT(n == static_cast<ssize_t>(str_array.size()));

	TokenVectorRef tokens_ref = std::make_shared<std::vector<Token>>();
	std::vector<Token> &tokens = *tokens_ref.get();

	const auto idx_array_r = idx_array.unchecked<1>();
	const auto len_array_r = len_array.unchecked<1>();

    for (ssize_t i = 0; i < n; i++) {
        Token t;

        t.id = p_vocab->add(str_array[i].cast<std::string_view>());

        t.idx = idx_array_r(i);
        t.len = len_array_r(i);
        t.pos = p_vocab->add_pos(pos_array[i].cast<std::string_view>());
        t.tag = p_vocab->add_tag(tag_array[i].cast<std::string_view>());
        tokens.push_back(t);
    }

	return tokens_ref;
}

TokenVectorRef unpack_tokens(
	const VocabularyRef &p_vocab,
	const py::dict &p_tokens) {
	return _unpack_tokens(p_vocab, p_tokens);
}

TokenVectorRef unpack_tokens(
	const QueryVocabularyRef &p_vocab,
	const py::dict &p_tokens) {
	return _unpack_tokens(p_vocab, p_tokens);
}

std::vector<StaticEmbeddingRef> QueryVocabulary::get_compiled_embeddings(const size_t p_embedding_index) const {
	const std::vector<EmbeddingRef> embeddings = {
		m_vocab->embedding_manager()->get_compiled(p_embedding_index),
		embedding_manager()->get_compiled(p_embedding_index)
	};

	std::vector<StaticEmbeddingRef> static_embeddings;
	static_embeddings.resize(embeddings.size());
	std::transform(embeddings.begin(), embeddings.end(), static_embeddings.begin(), [] (auto x) {
		return std::static_pointer_cast<StaticEmbedding>(x);
	});

	return static_embeddings;
}


void Frequencies::compute_tf_idf() {
	if (m_tf_idf_valid) {
		return;
	}
	xt::pytensor<float, 1> idf;
	idf.resize({static_cast<ssize_t>(m_tf.shape(0))});
	idf = xt::log(m_n_docs / xt::cast<float>(1 + m_df));
	m_tf_idf = m_tf * idf;
	m_tf_idf_valid = true;
}

Frequencies::Frequencies(const VocabularyRef &p_vocab) :
	m_vocab(p_vocab), m_n_docs(0) {

	const ssize_t size = static_cast<ssize_t>(p_vocab->size());

	m_tf.resize({size});
	m_tf.fill(0);

	m_df.resize({size});
	m_df.fill(0);

	m_tf_idf_valid = false;
}

void Frequencies::add(
	const DocumentRef &p_doc,
	const SliceStrategyRef &p_slice_strategy) {

	const SliceStrategy &slice_strategy = *p_slice_strategy;

	const Token *tokens = p_doc->tokens_vector()->data();
	std::unordered_set<token_t> seen;
	seen.reserve(slice_strategy.window_size);

	const auto spans = p_doc->spans(slice_strategy.level);

	spans->iterate(slice_strategy,
		[this, tokens, &seen] (const size_t slice_id, const size_t offset, const size_t len_s) {

			seen.clear();
			for (size_t i = 0; i < len_s; i++) {
				const auto id = tokens[offset + i].id;
				m_tf(id) += 1;
				if (seen.find(id) == seen.end()) {
					seen.insert(id);
					m_df(id) += 1;
				}
			}

			m_n_docs += 1;
			return true;

		});
}

/*xt::pytensor<float, 2> Frequencies::bow(
	const TokenContainerRef &p_container,
	const SliceStrategyRef &p_slice_strategy) {

	const SliceStrategy &slice_strategy = *p_slice_strategy;
	const auto spans = p_container->spans(slice_strategy.level);

	const size_t n_slices = spans->size();
	xt::pytensor<float, 2> data({n_slices, m_tf.shape(0)}, 0.0f);

	return data;
}*/
