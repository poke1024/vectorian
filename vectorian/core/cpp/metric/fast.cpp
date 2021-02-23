#include "metric/fast.h"
#include "scores/fast.h"
#include "query.h"
#include "match/matcher_impl.h"
#include "alignment/wmd.h"

#include <fstream>

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
		const QueryRef &, const Slice &slice, const int len_s, const int len_t) const {

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

	struct RefToken {
		wvec_t word_id;
		Index i; // index in s or t
		int8_t j; // 0 for s, 1 for t
	};

	const WMDOptions m_options;
	WMD<Index, RefToken> m_wmd;

	float m_score;

public:
	RelaxedWordMoversDistance(
		const bool p_normalize_bow,
		const bool p_symmetric,
		const bool p_one_target) :

		m_options(WMDOptions{
			p_normalize_bow,
			p_symmetric,
			p_one_target
		}) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_wmd.resize(max_len_s, max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline void operator()(
		const QueryRef &p_query, const Slice &slice, const int len_s, const int len_t) {

		const bool pos_tag_aware = p_query->is_pos_tag_aware();
		if (pos_tag_aware) {
			throw std::runtime_error(
				"tag weights and pos penalties are not yet supported for WMD");
		}

		const auto &enc = slice.encoder();

		const int vocabulary_size = m_wmd.init(
			slice, [&enc] (const auto &t) {
				return enc.to_embedding(t);
			}, len_s, len_t, m_options);

		if (vocabulary_size == 0) {
			m_score = 0.0f;
			return;
		}

		m_wmd.compute_dist(
			m_wmd.m_doc[0], m_wmd.m_doc[1], len_s, len_t,
			vocabulary_size,
			[&slice] (int i, int j) -> float {
				return slice.similarity(i, j);
			});

		std::ofstream outfile;
		outfile.open("/Users/arbeit/Desktop/debug_wmd.txt", std::ios_base::app);
		m_wmd.print_debug(p_query, slice, len_s, len_t, vocabulary_size, outfile);

		//p_query->t_tokens_pos_weights();

		m_score = 1.0f - m_wmd.wmd_relaxed(
			m_wmd.m_doc[0], m_wmd.m_doc[1], len_s, len_t,
			m_wmd.m_dist.data(),
			vocabulary_size,
			m_options);

		if (!pos_tag_aware) {
			m_score *= len_t;
		}

		outfile << "score: " << m_score << "\n";
		outfile << "\n";
	}

	inline float score() const {
		return m_score;
	}

	inline const std::vector<Index> &match() const {
		return m_wmd.match();
	}

	inline std::vector<Index> &mutable_match() {
		return m_wmd.match();
	}
};

template<typename Scores, typename Aligner>
MatcherRef make_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const std::vector<Scores> &scores,
	const Aligner &p_aligner) {

	if (p_query->bidirectional()) {
		return std::make_shared<MatcherImpl<Scores, Aligner, TokenIdEncoder, true>>(
			p_query, p_document, p_metric, p_aligner, scores);
	} else {
		return std::make_shared<MatcherImpl<Scores, Aligner, TokenIdEncoder, false>>(
			p_query, p_document, p_metric, p_aligner, scores);
	}
}

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

		return make_matcher(
			p_query, p_document, p_metric, scores,
			WatermanSmithBeyer<int16_t>(gap_cost, zero));

	} else if (algorithm == "rwmd") {

		bool normalize_bow = true;
		bool symmetric = true;
		bool one_target = true;

		if (args.contains("normalize_bow")) {
			normalize_bow = args["normalize_bow"].cast<bool>();
		}
		if (args.contains("symmetric")) {
			symmetric = args["symmetric"].cast<bool>();
		}
		if (args.contains("one_target")) {
			one_target = args["one_target"].cast<bool>();
		}

		return make_matcher(
			p_query, p_document, p_metric, scores,
			RelaxedWordMoversDistance<int16_t>(
				normalize_bow, symmetric, one_target));

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
