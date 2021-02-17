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

	virtual token_t lookup(const std::string &p_token) const {
		return -1;
	}

	virtual MetricRef create_metric(
		const TokenIdArray &p_vocabulary_to_embedding,
		const std::string &p_embedding_similarity,
		const std::string &p_needle_text,
		const std::vector<Token> &p_needle,
		float p_pos_mismatch_penalty,
		float p_similarity_falloff,
		float p_similarity_threshold,
		const POSWMap &p_pos_weights) = 0;

	const std::string &name() const {
		return m_name;
	}
};

#endif // __VECTORIAN_EMBEDDING_H__
