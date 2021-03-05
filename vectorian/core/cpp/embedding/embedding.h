#ifndef __VECTORIAN_EMBEDDING_H__
#define __VECTORIAN_EMBEDDING_H__

#include "common.h"
#include "sim.h"

class VocabularyToEmbedding;

class Embedding : public std::enable_shared_from_this<Embedding> {
	const std::string m_name;

public:
	Embedding(const std::string &p_name) : m_name(p_name) {
	}

	virtual ~Embedding() {
	}

	virtual void update_map(
		std::vector<token_t> &p_map,
		const std::vector<std::string> &p_tokens,
		const size_t p_offset) const = 0;

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const VocabularyToEmbedding &p_vocabulary_to_embedding) = 0;

	const std::string &name() const {
		return m_name;
	}
};

#endif // __VECTORIAN_EMBEDDING_H__
