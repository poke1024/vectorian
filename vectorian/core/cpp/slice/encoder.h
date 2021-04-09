#ifndef __VECTORIAN_TOKEN_ID_ENCODER_H__
#define __VECTORIAN_TOKEN_ID_ENCODER_H__

enum Dependency {
	NONE, // only depends on token id
	TAGS, // depends on tagging, e.g. POS tags
	POSITION // fully contextual
};

class StaticEmbeddingTokenIdEncoder {
public:
	inline wvec_t to_embedding(
		const int p_src,
		const size_t p_index,
		const Token &p_token) const {

		return p_token.id;
	}
};

class ContextualEmbeddingTokenIdEncoder {
public:
	inline wvec_t to_embedding(
		const int p_src,
		const size_t p_index,
		const Token &p_token) const {

		return p_src * 1000 + p_index; // FIXME
	}
};

class TokenIdPosEncoder {
	const size_t m_npos;

public:
	inline TokenIdPosEncoder(const size_t p_npos) : m_npos(p_npos) {
	}

	inline wvec_t to_embedding(
		const int p_src,
		const size_t p_index,
		const Token &p_token) const {

		return p_token.id * m_npos + p_token.pos;
	}
};

class TokenIdTagEncoder {
	const size_t m_ntag;

public:
	inline TokenIdTagEncoder(const size_t p_ntag) : m_ntag(p_ntag) {
	}

	inline wvec_t to_embedding(
		const int p_src,
		const size_t p_index,
		const Token &p_token) const {

		return p_token.id * m_ntag + p_token.tag;
	}
};

#endif // __VECTORIAN_TOKEN_ID_ENCODER_H__
