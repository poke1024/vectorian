#ifndef __VECTORIAN_EMBEDDING_H__
#define __VECTORIAN_EMBEDDING_H__

#include "common.h"
#include "sim.h"

class Embedding : public std::enable_shared_from_this<Embedding> {
	const std::string m_name;

public:
	Embedding(const std::string &p_name) : m_name(p_name) {
	}

	virtual ~Embedding() {
	}

	virtual token_t lookup(const std::string &p_token) const {
		return -1;
	}

	//virtual const std::vector<std::string> &tokens() const = 0;

	virtual MetricRef create_metric(
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<MappedTokenIdArray> &p_vocabulary_to_embedding,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle) = 0;

	const std::string &name() const {
		return m_name;
	}
};

#endif // __VECTORIAN_EMBEDDING_H__
