#include "common.h"
#include "alignment/wmd.h"

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
				return slice.similarity(i, j);
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
		const QueryRef &p_query,
		const Slice &slice,
		const int len_s,
		const int len_t) {

		const bool pos_tag_aware = slice.similarity_depends_on_pos();
		const auto &enc = slice.encoder();
		const float max_cost = m_options.normalize_bow ?
			1.0f : slice.max_sum_of_similarities();

		if (pos_tag_aware) {
			// perform WMD on a vocabulary
			// built from (token id, pos tag).

			m_score = m_wmd_tagged.relaxed(
				slice, len_s, len_t,
				[&enc] (const auto &t) {
					return TaggedTokenId{
						enc.to_embedding(t),
						t.tag
					};
				},
				m_options,
				max_cost);
		} else {
			// perform WMD on a vocabulary
			// built from token ids.

			m_score = m_wmd.relaxed(
				slice, len_s, len_t,
				[&enc] (const auto &t) {
					return enc.to_embedding(t);
				},
				m_options,
				max_cost);
		}
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


template<typename SliceFactory>
MatcherRef create_alignment_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const py::dict &p_alignment_def,
	const SliceFactory &p_factory) {

	// FIXME support different alignment algorithms here.

	std::string algorithm;
	if (p_alignment_def.contains("algorithm")) {
		algorithm = p_alignment_def["algorithm"].cast<py::str>();
	} else {
		algorithm = "wsb"; // default
	}

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
			WatermanSmithBeyer<int16_t>(gap_cost, zero));

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
			RelaxedWordMoversDistance<int16_t>(
				normalize_bow, symmetric, one_target));

	} else {

		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
}
