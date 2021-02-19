#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "match/matcher_impl.h"

template<typename Index>
class WatermanSmithBeyer {
	std::shared_ptr<Aligner<Index, float>> m_aligner;
	const std::vector<float> m_gap_cost;
	const float m_smith_waterman_zero;

public:
	WatermanSmithBeyer(
		const std::vector<float> &p_gap_cost,
		float p_zero=0.5) :

		m_gap_cost(p_gap_cost),
		m_smith_waterman_zero(p_zero) {

		PPK_ASSERT(m_gap_cost.size() >= 1);
	}

	void init(Index max_len_s, Index max_len_t) {
		m_aligner = std::make_shared<Aligner<Index, float>>(
			max_len_s, max_len_t);
	}

	float gap_cost(size_t len) const {
		return m_gap_cost[
			std::min(len, m_gap_cost.size() - 1)];
	}

	template<typename SentenceScores>
	inline void operator()(
		const SentenceScores &scores, int len_s, int len_t) const {

		m_aligner->waterman_smith_beyer(
			scores,
			[this] (size_t i) -> float {
				return m_gap_cost[
					std::min(i, m_gap_cost.size() - 1)];
			},
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

	py::gil_scoped_acquire acquire;

	const py::dict args = p_query->alignment_algorithm();

	std::string algorithm;
	if (args.contains("algorithm")) {
		algorithm = args["algorithm"].cast<py::str>();
	} else {
		algorithm = "wsb"; // default
	}

	if (algorithm == "wsb") {
		float zero = 0.5;
		if (args.contains("zero")) {
			zero = args["zero"].cast<float>();
		}

		std::vector<float> gap_cost;
		if (args.contains("gap")) {
			auto cost = args["gap"].cast<py::array_t<float>>();
			auto r = cost.unchecked<1>();
			const ssize_t n = r.shape(0);
			gap_cost.resize(n);
			for (ssize_t i = 0; i < n; i++) {
				gap_cost[i] = r(i);
			}
		}
		if (gap_cost.empty()) {
			gap_cost.push_back(std::numeric_limits<float>::infinity());
		}

		return std::make_shared<MatcherImpl<Scores, WatermanSmithBeyer<int16_t>>>(
			p_query, p_document, p_metric, WatermanSmithBeyer<int16_t>(gap_cost, zero), scores);
	} else {
		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
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
