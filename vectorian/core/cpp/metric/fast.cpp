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
				p_query->max_weighted_score());
		} else {
			// perform WMD on a vocabulary
			// built from token ids.

			m_score = m_wmd.relaxed(
				slice, len_s, len_t,
				[&enc] (const auto &t) {
					return enc.to_embedding(t);
				},
				m_options,
				p_query->max_weighted_score());
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

template<typename SliceFactory, typename Aligner>
MatcherRef make_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const SliceFactory &p_factory,
	const Aligner &p_aligner) {

	if (p_query->bidirectional()) {
		return std::make_shared<MatcherImpl<SliceFactory, Aligner, true>>(
			p_query, p_document, p_metric, p_aligner, p_factory);
	} else {
		return std::make_shared<MatcherImpl<SliceFactory, Aligner, false>>(
			p_query, p_document, p_metric, p_aligner, p_factory);
	}
}

template<typename SliceFactory>
MatcherRef create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const SliceFactory &p_factory) {

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
			p_query, p_document, p_metric, p_factory,
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
			p_query, p_document, p_metric, p_factory,
			RelaxedWordMoversDistance<int16_t>(
				normalize_bow, symmetric, one_target));

	} else {

		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
}

template<typename MakeSlice>
class FactoryGenerator {
	const MakeSlice m_make_slice;

public:
	typedef typename std::invoke_result<
		MakeSlice,
		const TokenSpan&,
		const TokenSpan&>::type Slice;

	FactoryGenerator(const MakeSlice &make_slice) :
		m_make_slice(make_slice) {
	}

	SliceFactory<MakeSlice> create(
		const DocumentRef &p_document) const {

		return SliceFactory(m_make_slice);
	}

	FilteredSliceFactory<SliceFactory<MakeSlice>> create_filtered(
		const DocumentRef &p_document,
		const TokenFilter &p_token_filter) const {

		return FilteredSliceFactory(
			create(p_document),
			p_document, p_token_filter);
	}
};

MatcherRef FastMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	const auto metric = std::dynamic_pointer_cast<FastMetric>(shared_from_this());

	const auto &token_filter = p_query->token_filter();

	const auto make_fast_slice = [metric] (
		const TokenSpan &s,
		const TokenSpan &t) {

        return FastSlice(metric, s, t);
	};

	const auto make_tag_weighted_slice = [metric] (
		const TokenSpan &s,
		const TokenSpan &t) {

		return TagWeightedSlice(
			FastSlice(metric, s, t),
			metric->modifiers());
	};

	FactoryGenerator gen(make_fast_slice);

	return ::create_matcher(
		p_query, p_document, metric,
		gen.create_filtered(p_document, token_filter));
}

const std::string &FastMetric::name() const {
	return m_embedding->name();
}
