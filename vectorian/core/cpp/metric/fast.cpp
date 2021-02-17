#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "matcher_impl.h"

struct CombineSum {
	inline float operator()(float x, float y) const {
		return x + y;
	}
};

struct CombineMin {
	inline float operator()(float x, float y) const {
		return std::min(x, y);
	}
};

struct CombineMax {
	inline float operator()(float x, float y) const {
		return std::max(x, y);
	}
};

template<typename Scores>
MatcherRef create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores) {

	return std::make_shared<MatcherImpl<Scores>>(
		p_query, p_document, p_metric, scores);
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
