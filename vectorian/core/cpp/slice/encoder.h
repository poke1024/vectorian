#ifndef __VECTORIAN_TOKEN_ID_ENCODER_H__
#define __VECTORIAN_TOKEN_ID_ENCODER_H__

class TokenIdEncoder {
public:
	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id;
	}
};

class TokenIdPosEncoder {
	const size_t m_npos;

public:
	inline TokenIdPosEncoder(const size_t p_npos) : m_npos(p_npos) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_npos + p_token.pos;
	}
};

class TokenIdTagEncoder {
	const size_t m_ntag;

public:
	inline TokenIdTagEncoder(const size_t p_ntag) : m_ntag(p_ntag) {
	}

	inline wvec_t to_embedding(const Token &p_token) const {
		return p_token.id * m_ntag + p_token.tag;
	}
};

#endif // __VECTORIAN_TOKEN_ID_ENCODER_H__
