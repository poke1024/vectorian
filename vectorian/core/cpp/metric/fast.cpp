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

	inline float gap_cost(size_t len) const {
		return m_gap_cost[
			std::min(len, m_gap_cost.size() - 1)];
	}

	template<typename Slice>
	inline void operator()(
		const Slice &slice, int len_s, int len_t) const {

		m_aligner->waterman_smith_beyer(
			[&slice] (int i, int j) -> float {
				return slice.score(i, j);
			},
			[this] (size_t len) -> float {
				return this->gap_cost(len);
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

template<typename Index>
class RelaxedWordMoversDistance {
	//EMDRelaxedCache m_cache;
	float m_score;
	std::vector<Index> m_match;

public:
	RelaxedWordMoversDistance() {
	}

	void init(Index max_len_s, Index max_len_t) {
		//m_cache.allocate(std::max(max_len_s, max_len_t));
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline void operator()(
		const Slice &slice, int len_s, int len_t) const {

		/*
		s are the corpus tokens, t are the query tokens.

		we build a new vocab that corresponds to w[i]
		sv[i] maps vocab item i to global token id (or -1 if not in s)
		tv[i] maps vocab item i to query token pos (or -1 if not in t)

		def sim(i, j):
			u = sv[i]
			v = tv[j]
			if (u >= 0 && v >= 0)
				slice.score(u, v)
			else:
				sim(j, i)

		those tokens in t that also occur in s:
			have distance 0 and share the same id.
		those tokens in t that do not occur in s:
			get new ids. for score lookup, we map those ids to the query token pos.
		*/

		// sort tuples (token_id, s_or_t, index)
		//std::sort();

		// w1: normalized bow for s
		// w2: normalized bow for t

		// size: size of vocabulary needed for this problem

		// dist: word distances in vocabulary

		//emd_relaxed(w1, w2, dist, size, cache);

		/*for (int i = 0; i < len_s; i++) {
			boilerplate[i] = i;
		}

		for (size_t i = 0; i < len_s; i++) {
			std::sort(
				boilerplate,
				boilerplate + len_s,
				[&] (const int a, const int b) {
					return scores(i, a) < scores(i, b);
				});

		m_score = emd_relaxed();

		m_aligner->waterman_smith_beyer(
			scores,
			[this] (size_t len) -> float {
				return this->gap_cost(len);
			},
			len_s,
			len_t,
			m_smith_waterman_zero);*/
	}

	inline float score() const {
		return m_score;
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &mutable_match() {
		return m_match;
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
