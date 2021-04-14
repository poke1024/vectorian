#include "embedding/static.h"
#include "metric/static.h"
#include "query.h"

Needle::Needle(
	const QueryRef &p_query) :

	m_needle(p_query->tokens_vector()) {

	const auto &needle = *m_needle;

	m_token_ids.resize({needle.size()});
	for (size_t i = 0; i < needle.size(); i++) {
		m_token_ids(i) = needle[i].id;
	}
}

StaticEmbedding::StaticEmbedding(
	py::object p_embedding_factory,
	py::list p_tokens) :

	Embedding(p_embedding_factory.attr("name").cast<std::string>()),
	m_size(0) {

	m_vectors = p_embedding_factory.attr("get_embeddings")(p_tokens);
	m_size = m_vectors.attr("size").cast<size_t>();
}

