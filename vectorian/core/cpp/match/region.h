#ifndef __VECTORIAN_REGION_H__
#define __VECTORIAN_REGION_H__

#include "common.h"
#include "vocabulary.h"
#include "match.h"

class Region {
	const Slice m_s;
	const float m_mismatch_penalty;

public:
	Region(
		const Slice &p_s,
		float p_mismatch_penalty = 0.0f):

		m_s(p_s),
		m_mismatch_penalty(p_mismatch_penalty) {
	}

	virtual ~Region() {
	}

	py::tuple s() const {
		return m_s.to_py();
	}

	float mismatch_penalty() const {
		return m_mismatch_penalty;
	}

	virtual bool is_matched() const {
		return false;
	}
};

typedef std::shared_ptr<Region> RegionRef;


struct TokenRef {
	TokenVectorRef tokens;
	int32_t index;

	const Token *operator->() const {
		return &tokens->at(index);
	}
};

class MatchedRegion : public Region {
public:
	class QueryToken {
		const QueryVocabularyRef m_vocab;
		const TokenRef m_token;
		const Slice m_slice;

	public:
		QueryToken(
			const QueryVocabularyRef &p_vocab,
			const TokenRef &p_token,
			const Slice &p_slice) :
			m_vocab(p_vocab), m_token(p_token), m_slice(p_slice) {
		}

		const size_t index() const {
			return m_token.index;
		}

		py::tuple slice() const {
			return m_slice.to_py();
		}

		const std::string_view &pos() const {
			return m_vocab->pos_str(m_token->pos);
		}
	};

	typedef std::shared_ptr<QueryToken> QueryTokenRef;


	class HalfEdge {
		const QueryVocabularyRef m_vocab;
		const Weight m_weight;
		const Slice m_slice;
		const TokenRef m_token;
		const std::string m_metric;

	public:
		HalfEdge(
			const QueryVocabularyRef &p_vocab,
			const Weight &p_weight,
			const Slice &p_slice,
			const TokenRef &p_token,
			const std::string &p_metric) :

			m_vocab(p_vocab),
			m_weight(p_weight),
			m_slice(p_slice),
			m_token(p_token),
			m_metric(p_metric) {
		}

		float flow() const {
			return m_weight.flow;
		}

		float distance() const {
			return m_weight.distance;
		}

		QueryTokenRef token() const {
			return std::make_shared<QueryToken>(
				m_vocab, m_token, m_slice);
		}

		const std::string &metric() const {
			return m_metric;
		}
	};

	typedef std::shared_ptr<HalfEdge> HalfEdgeRef;

private:
	const QueryVocabularyRef m_vocab;
	const TokenRef m_s_token;
	std::vector<HalfEdgeRef> m_edges;

public:
	MatchedRegion(
		const QueryVocabularyRef &p_vocab,
		const Slice &p_s,
		const TokenRef &p_s_token,
		std::vector<HalfEdgeRef> &&p_edges) :

		Region(p_s),
		m_vocab(p_vocab),
		m_s_token(p_s_token),
		m_edges(p_edges) {
	}

	virtual bool is_matched() const {
		return true;
	}

	size_t num_edges() const {
		return m_edges.size();
	}

	HalfEdgeRef edge(const size_t p_index) const {
		return m_edges.at(p_index);
	}

	const std::string_view &pos_s() const {
		return m_vocab->pos_str(m_s_token->pos);
	}
};

typedef std::shared_ptr<MatchedRegion> MatchedRegionRef;

#endif // __VECTORIAN_REGION_H__
