#include "common.h"
#include "alignment/wmd.h"
#include "alignment/wrd.h"

template<typename Slice>
inline float reference_score(
	const QueryRef &p_query,
	const Slice &p_slice,
	const float p_matched,
	const float p_unmatched) {

	// m_matched_weight == 0 indicates that there
	// is no higher relevance of matched content than
	// unmatched content, both are weighted equal (see
	// maximum_internal_score()).

	const float total_score = p_slice.max_sum_of_similarities();

	const float unmatched_weight = std::pow(
		(total_score - p_matched) / total_score,
		p_query->submatch_weight());

	const float reference_score =
		p_matched +
		unmatched_weight * (total_score - p_matched);

	return reference_score;
}

template<typename Slice, typename Index>
inline float normalized_score(
	const QueryRef &p_query,
	const Slice &p_slice,
	const float p_raw_score,
	const std::vector<Index> &p_match) {

	//return p_raw_score / p_slice.len_t(); // FIXME

	// unboosted version would be:
	// return p_raw_score / m_total_score;

	// a final boosting step allowing matched content
	// more weight than unmatched content.

	const size_t n = p_match.size();

	float matched_score = 0.0f;
	float unmatched_score = 0.0f;

	for (size_t i = 0; i < n; i++) {

		const float s = p_slice.max_similarity_for_t(i);

		if (p_match[i] < 0) {
			unmatched_score += s;
		} else {
			matched_score += s;
		}
	}

	return p_raw_score / reference_score(
		p_query, p_slice, matched_score, unmatched_score);
}

template<typename Index>
class WatermanSmithBeyer {
	std::shared_ptr<Aligner<Index, float>> m_aligner;
	const std::vector<float> m_gap_cost;
	const float m_smith_waterman_zero;

	template<typename Slice>
	inline void compute(
		const QueryRef &, const Slice &slice) const {

		m_aligner->waterman_smith_beyer(
			[&slice] (int i, int j) -> float {
				return slice.similarity(i, j);
			},
			[this] (size_t len) -> float {
				return this->gap_cost(len);
			},
			slice.len_s(),
			slice.len_t(),
			m_smith_waterman_zero);
	}

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
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) const {

		compute(p_matcher->query(), p_slice);

		// flow = flow_factory->create_1_to_1(m_aligner->match());

		const float score = normalized_score(
			p_matcher->query(), p_slice, m_aligner->score(), m_aligner->match());

		if (score > p_result_set->worst_score()) {
			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				p_result_set->flow_factory()->create_1_to_1(m_aligner->match()),
				score);
		} else {
			return MatchRef();
		}
	}

	template<typename SliceFactory>
	class ScoreComputer {
		const SliceFactory m_factory;

	public:
		inline ScoreComputer(const SliceFactory &p_factory) : m_factory(p_factory) {
		}

		void operator()(const MatchRef &p_match) const {
			auto *flow = static_cast<OneToOneFlow<Index>*>(p_match->flow().get());
			auto &mapping = flow->mapping();

		    const auto match_slice = p_match->slice();
	        const auto token_at = match_slice.idx;

	        Index end = 0;
	        for (auto m : mapping) {
	            end = std::max(end, m.target);
	        }

			const Token *s_tokens = p_match->document()->tokens()->data();
			const auto &t_tokens = p_match->query()->tokens();

	        const auto slice = m_factory.create_slice(
	            0,
	            TokenSpan{s_tokens + token_at, match_slice.len},
	            TokenSpan{t_tokens->data(), static_cast<int32_t>(t_tokens->size())});

	        Index i = 0;
	        for (auto &m : mapping) {
	            if (m.target >= 0) {
	                m.similarity = slice.unmodified_similarity(m.target, i);
	                m.weight = 1.0f; // FIXME; was: slice.weight(m, i)
	            } else {
	                m.similarity = 0.0f;
	                m.weight = 0.0f;
	            }
	            i++;
	        }
		}
	};

	template<typename SliceFactory>
	static ScoreComputer<SliceFactory> create_score_computer(const SliceFactory &p_factory) {
		return ScoreComputer<SliceFactory>(p_factory);
	}
};

class NoScoreComputer {
public:
	inline void operator()(const MatchRef &) const {
		// no op.
	}
};

template<typename Index>
class RelaxedWordMoversDistance {

	struct TaggedTokenId {
		token_t token;
		int8_t tag;

		inline bool operator==(const TaggedTokenId &t) const {
			return token == t.token && tag == t.tag;
		}

		inline bool operator!=(const TaggedTokenId &t) const {
			return !(*this == t);
		}

		inline bool operator<(const TaggedTokenId &t) const {
			if (token < t.token) {
				return true;
			} else if (token == t.token) {
				return tag < t.tag;
			} else {
				return false;
			}
		}
	};

	const WMDOptions m_options;
	WMD<Index, token_t> m_wmd;
	WMD<Index, TaggedTokenId> m_wmd_tagged;

	struct Result {
		float score;
		const WMDBase<Index> &wmd;
	};

	template<typename Slice>
	inline Result compute(
		const QueryRef &p_query,
		const Slice &slice) {

		const bool pos_tag_aware = slice.similarity_depends_on_pos();
		const auto &enc = slice.encoder();
		const float max_cost = m_options.normalize_bow ?
			1.0f : slice.max_sum_of_similarities();

		if (pos_tag_aware) {
			// perform WMD on a vocabulary
			// built from (token id, pos tag).

			const float score = m_wmd_tagged.relaxed(
				slice,
				[&enc] (const auto &t) {
					return TaggedTokenId{
						enc.to_embedding(t),
						t.tag
					};
				},
				m_options,
				max_cost);

			return Result{score, m_wmd_tagged};

		} else {
			// perform WMD on a vocabulary
			// built from token ids.

			const float score = m_wmd.relaxed(
				slice,
				[&enc] (const auto &t) {
					return enc.to_embedding(t);
				},
				m_options,
				max_cost);

			return Result{score, m_wmd};
		}
	}

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
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) {

		const auto r = compute(p_matcher->query(), p_slice);

		const float score = normalized_score(
			p_matcher->query(), p_slice, r.score, r.wmd.match());

		if (score > p_result_set->worst_score()) {
			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				p_result_set->flow_factory()->create_1_to_1(r.wmd.match()),
				score);
		} else {
			return MatchRef();
		}
	}
};

template<typename Index>
class WordRotatorsDistance {
	WRD<Index> m_wrd;

public:
	WordRotatorsDistance() {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_wrd.resize(max_len_s, max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) {

		const float score0 = m_wrd.compute(
			p_matcher->query(), p_slice);

		const float score = normalized_score(
			p_matcher->query(), p_slice, score0, m_wrd.match());

		if (score > p_result_set->worst_score()) {
			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				p_result_set->flow_factory()->create_1_to_1(m_wrd.match()),
				score);
		} else {
			return MatchRef();
		}
	}
};

inline std::string get_alignment_algorithm(
	const py::dict &p_alignment_def) {

	if (p_alignment_def.contains("algorithm")) {
		return p_alignment_def["algorithm"].cast<py::str>();
	} else {
		return "wsb"; // default
	}
}

inline MatcherOptions create_alignment_matcher_options(
	const py::dict &p_alignment_def) {

	const std::string algorithm = get_alignment_algorithm(p_alignment_def);
	return MatcherOptions{algorithm == "wrd"};
}

template<typename Index, typename SliceFactory>
MatcherRef create_alignment_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const py::dict &p_alignment_def,
	const SliceFactory &p_factory) {

	const std::string algorithm = get_alignment_algorithm(p_alignment_def);

	if (algorithm == "wsb") {
		float zero = 0.5;
		if (p_alignment_def.contains("zero")) {
			zero = p_alignment_def["zero"].cast<float>();
		}

		std::vector<float> gap_cost;
		if (p_alignment_def.contains("gap")) {
			auto cost = p_alignment_def["gap"].cast<py::array_t<float>>();
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
			p_query, p_document, p_metric, p_factory,
			std::move(WatermanSmithBeyer<Index>(gap_cost, zero)),
			WatermanSmithBeyer<Index>::create_score_computer(p_factory));

	} else if (algorithm == "rwmd") {

		bool normalize_bow = true;
		bool symmetric = true;
		bool one_target = true;

		if (p_alignment_def.contains("normalize_bow")) {
			normalize_bow = p_alignment_def["normalize_bow"].cast<bool>();
		}
		if (p_alignment_def.contains("symmetric")) {
			symmetric = p_alignment_def["symmetric"].cast<bool>();
		}
		if (p_alignment_def.contains("one_target")) {
			one_target = p_alignment_def["one_target"].cast<bool>();
		}

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(RelaxedWordMoversDistance<Index>(
				normalize_bow, symmetric, one_target)),
			NoScoreComputer());

	} else if (algorithm == "wrd") {

		return make_matcher(p_query, p_document, p_metric, p_factory,
			std::move(WordRotatorsDistance<Index>()),
			NoScoreComputer());

	} else {

		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
}
