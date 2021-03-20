#ifndef __VECTORIAN_EMBEDDING_H__
#define __VECTORIAN_EMBEDDING_H__

#include "common.h"

class Embedding : public std::enable_shared_from_this<Embedding> {
	const std::string m_name;

public:
	Embedding(const std::string &p_name) : m_name(p_name) {
	}

	virtual ~Embedding() {
	}

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<EmbeddingRef> &p_embeddings) = 0;

	const std::string &name() const {
		return m_name;
	}
};

#endif // __VECTORIAN_EMBEDDING_H__
