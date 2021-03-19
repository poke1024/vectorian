#ifndef __VECTORIAN_FAST_EMBEDDING_H__
#define __VECTORIAN_FAST_EMBEDDING_H__

#include "common.h"
#include "embedding/embedding.h"
#include "embedding/vectors.h"
#include "utils.h"
#include <xtensor/xadapt.hpp>

class Needle {
	const QueryRef m_query;
	const TokenVectorRef m_needle;
	TokenIdArray m_token_ids;

public:
	Needle(const QueryRef &p_query);

	inline const QueryRef &query() const {
		return m_query;
	}

	inline const size_t size() const {
		return m_needle->size();
	}

	inline const TokenIdArray &token_ids() const {
		return m_token_ids;
	}
};

class StaticEmbedding : public Embedding {
	py::object m_vectors; // type: vectorian.embeddings.Vectors
	size_t m_size;

public:
	StaticEmbedding(
		py::object p_embedding_factory,
		py::list p_tokens);

	inline py::object &vectors() {
		return m_vectors;
	}

	py::object py_vectors() const {
		return m_vectors;
	}

	size_t size() const {
		return m_size;
	}

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<EmbeddingRef> &p_embeddings);
};

typedef std::shared_ptr<StaticEmbedding> StaticEmbeddingRef;

inline py::object pick_vectors(
	const std::vector<StaticEmbeddingRef> &p_embeddings,
	const size_t p_index,
	size_t &r_index) {

	size_t i = p_index;
	for (const auto &embedding : p_embeddings) {
		const auto &vectors = embedding->vectors();
		const size_t size = embedding->size();
		if (i < size) {
			r_index = i;
			return vectors;
		}
		i -= size;
	}

	std::ostringstream err;
	err << "pick_vectors: " << p_index << " > ";
	for (const auto &embedding : p_embeddings) {
		err << embedding->size();
		if (embedding != p_embeddings.back()) {
			err << " + ";
		}
	}
	throw std::runtime_error(err.str());
}

#endif // __VECTORIAN_FAST_EMBEDDING_H__
