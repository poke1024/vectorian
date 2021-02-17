#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "matcher_impl.h"

template<typename Index>
class WatermanSmithBeyer {
	std::shared_ptr<Aligner<Index, float>> m_aligner;
	MismatchPenaltyRef m_gap_cost;
	const float m_smith_waterman_zero;

public:
	WatermanSmithBeyer(
		const MismatchPenaltyRef &p_gap_cost,
		float p_zero=0.5) :

		m_gap_cost(p_gap_cost),
		m_smith_waterman_zero(p_zero) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_aligner = std::make_shared<Aligner<Index, float>>(
			max_len_s, max_len_t);
	}

	template<typename SentenceScores>
	inline void operator()(
		const SentenceScores &scores, int len_s, int len_t) const {

		m_aligner->waterman_smith_beyer(
			scores,
			*m_gap_cost,
			len_s,
			len_t,
			m_smith_waterman_zero);
	}

	inline float score() const {
		return m_aligner->score();
	}

	inline const std::vector<Index> &match() const {
		return m_aligner->match();
	}

	inline std::vector<Index> &mutable_match() {
		return m_aligner->mutable_match();
	}
};

template<typename Scores>
MatcherRef create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores) {

	// FIXME support different alignment algorithms here.

	auto gap_cost = std::make_shared<MismatchPenalty>(
		p_query->mismatch_length_penalty(),
		p_document->max_len_s());

	return std::make_shared<MatcherImpl<Scores, WatermanSmithBeyer<int16_t>>>(
		p_query, p_document, p_metric, WatermanSmithBeyer<int16_t>(gap_cost), scores);
}


MatcherRef FastMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	auto self = std::dynamic_pointer_cast<FastMetric>(shared_from_this());

	std::vector<FastScores> scores;
	scores.emplace_back(FastScores(p_query, p_document, self));

	return ::create_matcher(p_query, p_document, self, scores);
}

const std::string &FastMetric::name() const {
	return m_embedding->name();
}
