#include "metric/static.h"
#include "slice/static.h"
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

std::vector<float> parse_tag_weights(
	const QueryRef &p_query,
	py::dict tag_weights) {

	std::map<std::string, float> pos_weights;
	for (const auto &pw : tag_weights) {
		pos_weights[pw.first.cast<py::str>()] = pw.second.cast<float>();
	}

	const auto mapped_pos_weights =
		p_query->vocabulary()->mapped_pos_weights(pos_weights);
	const auto t_tokens = p_query->tokens();

	std::vector<float> t_tokens_pos_weights;
	t_tokens_pos_weights.reserve(t_tokens->size());
	for (size_t i = 0; i < t_tokens->size(); i++) {
		const Token &t = t_tokens->at(i);
		const auto w = mapped_pos_weights.find(t.tag);
		float s;
		if (w != mapped_pos_weights.end()) {
			s = w->second;
		} else {
			s = 1.0f;
		}
		t_tokens_pos_weights.push_back(s);
	}

	return t_tokens_pos_weights;
}

MatcherRef StaticEmbeddingMetric::create_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document) {

	py::gil_scoped_acquire acquire;

	const auto metric = std::dynamic_pointer_cast<StaticEmbeddingMetric>(shared_from_this());

	const auto &token_filter = p_query->token_filter();

	const std::string sentence_metric_kind =
		m_options["metric"].cast<py::str>();

	if (sentence_metric_kind == "alignment-isolated") {

		const auto make_fast_slice = [metric] (
			const TokenSpan &s,
			const TokenSpan &t) {

	        return StaticEmbeddingSlice(metric, s, t);
		};

		const FactoryGenerator gen(make_fast_slice);

		return ::create_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_document, token_filter));

	} else if (sentence_metric_kind == "alignment-tag-weighted") {

		TagWeightedOptions options;
		options.t_pos_weights = parse_tag_weights(p_query, m_options["tag_weights"]);
		options.pos_mismatch_penalty = m_options["pos_mismatch_penalty"].cast<float>();
		options.similarity_threshold = m_options["similarity_threshold"].cast<float>();

		float sum = 0.0f;
		for (float w : options.t_pos_weights) {
			sum += w;
		}
		options.t_pos_weights_sum = sum;

		const auto make_tag_weighted_slice = [metric, options] (
			const TokenSpan &s,
			const TokenSpan &t) {

			return TagWeightedSlice(
				StaticEmbeddingSlice(metric, s, t),
				options);
		};

		const FactoryGenerator gen(make_tag_weighted_slice);

		return ::create_matcher(
			p_query, p_document, metric, metric->alignment_def(),
			gen.create_filtered(p_document, token_filter));
	} else {

		std::ostringstream err;
		err << "unknown sentence metric type " << sentence_metric_kind;
		throw std::runtime_error(err.str());
	}
}

const std::string &StaticEmbeddingMetric::name() const {
	return m_embedding->name();
}
