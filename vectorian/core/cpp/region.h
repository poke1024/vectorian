#include "common.h"
#include "vocabulary.h"
#include "match.h"


class Region {
	const std::string m_s;
	const float m_mismatch_penalty;

public:
	Region(
		std::string &&s,
		float p_mismatch_penalty = 0.0f):

		m_s(s),
		m_mismatch_penalty(p_mismatch_penalty) {
	}

	virtual ~Region() {
	}

	py::bytes s() const {
		return m_s;
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
	const TokenScore m_score; // score between s and t
	const std::string m_t;

	const VocabularyRef m_vocab;
	const TokenRef m_s_token;
	const TokenRef m_t_token;

	const std::string m_metric;

public:
	MatchedRegion(
		const TokenScore &p_score,
		std::string &&s,
		std::string &&t,
		const VocabularyRef &p_vocab,
		const TokenRef &p_s_token,
		const TokenRef &p_t_token,
		const std::string &p_metric) :

		Region(std::move(s)),
		m_score(p_score),
		m_t(t),

		m_vocab(p_vocab),
		m_s_token(p_s_token),
		m_t_token(p_t_token),

		m_metric(p_metric) {
		}

		virtual bool is_matched() const {
			return true;
		}

		float similarity() const {
			return m_score.similarity;
		}

		float weight() const {
			return m_score.weight;
		}

		py::bytes t() const {
			return m_t;
		}

		py::bytes pos_s() const {
			return m_vocab->pos_str(m_s_token->pos);
		}

		py::bytes pos_t() const {
			return m_vocab->pos_str(m_t_token->pos);
		}

		py::bytes metric() const {
			return m_metric;
		}
};

typedef std::shared_ptr<MatchedRegion> MatchedRegionRef;
