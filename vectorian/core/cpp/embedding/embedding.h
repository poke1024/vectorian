#ifndef __VECTORIAN_EMBEDDING_H__
#define __VECTORIAN_EMBEDDING_H__

#include "common.h"

class TokenEmbedding : public std::enable_shared_from_this<TokenEmbedding> {
	const std::string m_name;

public:
	TokenEmbedding(const std::string &p_name) : m_name(p_name) {
	}

	virtual ~TokenEmbedding() {
	}

	const std::string &name() const {
		return m_name;
	}
};

#endif // __VECTORIAN_EMBEDDING_H__
