#include "common.h"
#include "alignment/wmd.h"
#include "alignment/wrd.h"

template<typename Slice>
inline float reference_score(
	const QueryRef &p_query,
	const Slice &p_slice,
	const MaximumScore &p_max) {

	// m_matched_weight == 0 indicates that there
	// is no higher relevance of matched content than
	// unmatched content, both are weighted equal (see
	// maximum_internal_score()).

	const float total_score = p_slice.max_sum_of_similarities();

	const float unmatched_weight = std::pow(
		(total_score - p_max.matched) / total_score,
		p_query->submatch_weight());

	const float reference_score =
		p_max.matched +
		unmatched_weight * (total_score - p_max.matched);

	return reference_score;
}

template<typename Index, typename Kernel>
class InjectiveAlignment {
protected:
	const char *m_callback_name;
	const Kernel m_kernel;
	std::shared_ptr<Aligner<Index, float>> m_aligner;
	size_t m_max_len_t;
	mutable InjectiveFlowRef<Index> m_cached_flow;

	template<typename Slice>
	void call_debug_hook(
		const QueryRef &p_query,
		const Slice &p_slice,
		const InjectiveFlowRef<Index> &p_flow,
		const float p_score) const {

		py::gil_scoped_acquire acquire;

		py::dict data;
		data["values"] = xt::pyarray<float>(m_aligner->value_matrix(
			p_slice.len_s(), p_slice.len_t()));
		data["traceback"] = xt::pyarray<float>(m_aligner->traceback_matrix(
			p_slice.len_s(), p_slice.len_t()));
		data["flow"] = p_flow->to_py();
		data["score"] = p_score;

		const auto callback = *p_query->debug_hook();
		callback(m_callback_name, data);
	}

public:
	InjectiveAlignment(
		const char *p_callback_name,
		const Kernel &p_kernel) :

		m_callback_name(p_callback_name),
		m_kernel(p_kernel),
		m_max_len_t(0) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_aligner = std::make_shared<Aligner<Index, float>>(
			max_len_s, max_len_t);
		m_max_len_t = max_len_t;
	}

	template<typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) const {

		if (!m_cached_flow) {
			m_cached_flow = p_result_set->flow_factory()->create_injective();
			m_cached_flow->reserve(m_max_len_t);
		}

		m_kernel(*m_aligner.get(), p_matcher->query(), p_slice, *m_cached_flow.get());

		const float score = m_aligner->score() / reference_score(
			p_matcher->query(), p_slice, m_cached_flow->max_score(p_slice));

		if (score > p_result_set->worst_score()) {
			const auto flow = m_cached_flow;
			m_cached_flow.reset();

			if (p_matcher->query()->debug_hook().has_value()) {
				call_debug_hook(p_matcher->query(), p_slice, flow, m_aligner->score());
			}

			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				flow,
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
			auto *flow = static_cast<InjectiveFlow<Index>*>(p_match->flow().get());
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
	            TokenSpan{s_tokens, token_at, match_slice.len},
	            TokenSpan{t_tokens->data(), 0, static_cast<int32_t>(t_tokens->size())});

	        Index i = 0;
	        for (auto &m : mapping) {
	            if (m.target >= 0) {
	                m.weight.flow = 1.0f;
	                m.weight.distance = 1.0f - slice.unmodified_similarity(m.target, i);
	            } else {
	                m.weight.flow = 0.0f;
	                m.weight.distance = 1.0f;
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

struct NeedlemanWunschKernel {
	const float m_gap_cost;

	template<typename Aligner, typename Slice, typename Flow>
	inline void operator()(
		Aligner &p_aligner,
		const QueryRef &,
		const Slice &p_slice,
		Flow &p_flow) const {

		p_aligner.needleman_wunsch(
			p_flow,
			[&p_slice] (auto i, auto j) -> float {
				return p_slice.similarity(i, j);
			},
			m_gap_cost,
			p_slice.len_s(),
			p_slice.len_t());
	}

	inline float gap_cost(size_t len) const {
		return m_gap_cost * len;
	}
};

template<typename Index>
class NeedlemanWunsch : public InjectiveAlignment<Index, NeedlemanWunschKernel> {
public:
	NeedlemanWunsch(
		const float p_gap_cost) :

		InjectiveAlignment<Index, NeedlemanWunschKernel>(
			"alignment/needleman-wunsch",
			NeedlemanWunschKernel{p_gap_cost}) {
	}

	inline float gap_cost(size_t len) const {
		return this->m_kernel.gap_cost(len);
	}
};

struct SmithWatermanKernel {
	const float m_gap_cost;
	const float m_zero;

	template<typename Aligner, typename Slice, typename Flow>
	inline void operator()(
		Aligner &p_aligner,
		const QueryRef &,
		const Slice &p_slice,
		Flow &p_flow) const {

		p_aligner.smith_waterman(
			p_flow,
			[&p_slice] (auto i, auto j) -> float {
				return p_slice.similarity(i, j);
			},
			m_gap_cost,
			p_slice.len_s(),
			p_slice.len_t(),
			m_zero);
	}

	inline float gap_cost(size_t len) const {
		return m_gap_cost * len;
	}
};

template<typename Index>
class SmithWaterman : public InjectiveAlignment<Index, SmithWatermanKernel> {
public:
	SmithWaterman(
		const float p_gap_cost,
		const float p_zero=0.5) :

		InjectiveAlignment<Index, SmithWatermanKernel>(
			"alignment/smith-waterman",
			SmithWatermanKernel{p_gap_cost, p_zero}) {
	}

	inline float gap_cost(size_t len) const {
		return this->m_kernel.gap_cost(len);
	}
};

struct WatermanSmithBeyerKernel {
	const std::vector<float> m_gap_cost;
	const float m_zero;

	template<typename Aligner, typename Slice, typename Flow>
	inline void operator()(
		Aligner &p_aligner,
		const QueryRef &,
		const Slice &p_slice,
		Flow &p_flow) const {

		p_aligner.waterman_smith_beyer(
			p_flow,
			[&p_slice] (auto i, auto j) -> float {
				return p_slice.similarity(i, j);
			},
			[this] (auto len) -> float {
				return this->gap_cost(len);
			},
			p_slice.len_s(),
			p_slice.len_t(),
			m_zero);
		}

	inline float gap_cost(size_t len) const {
		return m_gap_cost[
			std::min(len, m_gap_cost.size() - 1)];
	}
};

template<typename Index>
class WatermanSmithBeyer : public InjectiveAlignment<Index, WatermanSmithBeyerKernel> {
public:
	WatermanSmithBeyer(
		const std::vector<float> &p_gap_cost,
		const float p_zero=0.5) :

		InjectiveAlignment<Index, WatermanSmithBeyerKernel>(
			"alignment/waterman-smith-beyer",
			WatermanSmithBeyerKernel{p_gap_cost, p_zero}) {

		PPK_ASSERT(p_gap_cost.size() >= 1);
	}

	inline float gap_cost(size_t len) const {
		return this->m_kernel.gap_cost(len);
	}
};

class NoScoreComputer {
public:
	inline void operator()(const MatchRef &) const {
		// no op.
	}
};

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

template<typename Index>
class WordMoversDistance {
	const WMDOptions m_options;

	WMD<Index, token_t> m_wmd;
	WMD<Index, TaggedTokenId> m_wmd_tagged;

	template<typename Slice, typename Solver>
	inline WMDSolution<typename Solver::FlowRef> compute(
		const QueryRef &p_query,
		const Slice &p_slice,
		const Solver &p_solver) {

		const bool pos_tag_aware = p_slice.similarity_depends_on_pos();
		const auto &enc = p_slice.encoder();

		if (pos_tag_aware) {
			// perform WMD on a vocabulary
			// built from (token id, pos tag).

			return m_wmd_tagged(
				p_query,
				p_slice,
				[&enc] (const auto &t) {
					return TaggedTokenId{
						enc.to_embedding(t),
						t.tag
					};
				},
				m_options,
				p_solver);

		} else {
			// perform WMD on a vocabulary
			// built from token ids.

			return m_wmd(
				p_query,
				p_slice,
				[&enc] (const auto &t) {
					return enc.to_embedding(t);
				},
				m_options,
				p_solver);
		}
	}

	template<typename Slice, typename Solver>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set,
		const Solver &p_solver) {

		const auto r = compute(
			p_matcher->query(),
			p_slice,
			p_solver);

		if (!r.flow) {
			return MatchRef();
		}

		const float score = r.score / reference_score(
			p_matcher->query(), p_slice, r.flow->max_score(p_slice));

		if (p_matcher->query()->debug_hook().has_value()) {
			py::gil_scoped_acquire acquire;
			const auto callback = *p_matcher->query()->debug_hook();
			py::dict data;
			data["score"] = score;
			data["worst_score"] = p_result_set->worst_score();
			callback("alignment/word-movers-distance/make", data);
		}

		if (score > p_result_set->worst_score()) {
			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				r.flow,
				score);
		} else {
			return MatchRef();
		}
	}

public:
	WordMoversDistance(
		const WMDOptions &p_options) :

		m_options(p_options) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_wmd.allocate(max_len_s, max_len_t);
	}

	inline float gap_cost(size_t len) const {
		return 0;
	}

	template<typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) {

		const FlowFactoryRef<Index> flow_factory =
			p_result_set->flow_factory();

		if (m_options.relaxed) {
			return make_match(
				p_matcher,
				p_slice,
				p_result_set,
				typename AbstractWMD<Index>::RelaxedSolver(flow_factory));
		} else {
			return make_match(
				p_matcher,
				p_slice,
				p_result_set,
				typename AbstractWMD<Index>::FullSolver(flow_factory));
		}
	}
};

template<typename Index>
class WordRotatorsDistance {
	WRD<Index> m_wrd;
	WRDOptions m_options;

public:
	WordRotatorsDistance(
		const WRDOptions &p_options) :
		m_options(p_options) {
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

		const FlowFactoryRef<Index> flow_factory =
			p_result_set->flow_factory();

		const auto r = m_wrd.compute(
			p_matcher->query(), p_slice, flow_factory, m_options);

		const float score = r.score / reference_score(
			p_matcher->query(), p_slice, r.flow->max_score(p_slice));

		if (score > p_result_set->worst_score()) {
			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				r.flow,
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
	return MatcherOptions{algorithm == "word-rotators-distance"};
}

template<typename Index, typename SliceFactory>
MatcherRef create_alignment_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const py::dict &p_alignment_def,
	const MatcherOptions &p_matcher_options,
	const SliceFactory &p_factory) {

	const std::string algorithm = get_alignment_algorithm(p_alignment_def);

	if (algorithm == "waterman-smith-beyer") {
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

	} else if (algorithm == "smith-waterman") {

		float gap_cost = 0.0f;
		if (p_alignment_def.contains("gap_cost")) {
			gap_cost = p_alignment_def["gap_cost"].cast<float>();
		}

		float zero = 0.5f;
		if (p_alignment_def.contains("zero")) {
			zero = p_alignment_def["zero"].cast<float>();
		}

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(SmithWaterman<Index>(gap_cost, zero)),
			SmithWaterman<Index>::create_score_computer(p_factory));

	} else if (algorithm == "needleman-wunsch") {

		float gap_cost = 0.0f;
		if (p_alignment_def.contains("gap_cost")) {
			gap_cost = p_alignment_def["gap_cost"].cast<float>();
		}

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(NeedlemanWunsch<Index>(gap_cost)),
			NeedlemanWunsch<Index>::create_score_computer(p_factory));

	} else if (algorithm == "word-movers-distance") {

		bool relaxed = true;
		bool normalize_bow = true;
		bool symmetric = true;
		bool injective = true;
		float extra_mass_penalty = -1.0f;

		if (p_alignment_def.contains("relaxed")) {
			relaxed = p_alignment_def["relaxed"].cast<bool>();
		}
		if (p_alignment_def.contains("normalize_bow")) {
			normalize_bow = p_alignment_def["normalize_bow"].cast<bool>();
		}
		if (p_alignment_def.contains("symmetric")) {
			symmetric = p_alignment_def["symmetric"].cast<bool>();
		}
		if (p_alignment_def.contains("injective")) {
			injective = p_alignment_def["injective"].cast<bool>();
		}
		if (p_alignment_def.contains("extra_mass_penalty")) {
			extra_mass_penalty = p_alignment_def["extra_mass_penalty"].cast<float>();
		}

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(WordMoversDistance<Index>(WMDOptions{
				relaxed, normalize_bow, symmetric, injective, pyemd, extra_mass_penalty})),
			NoScoreComputer());

	} else if (algorithm == "word-rotators-distance") {

		PPK_ASSERT(p_matcher_options.needs_magnitudes);

		bool normalize_magnitudes = true;
		if (p_alignment_def.contains("normalize_magnitudes")) {
			normalize_magnitudes = p_alignment_def["normalize_magnitudes"].cast<bool>();
		}

		float extra_mass_penalty = -1.0f;
		if (p_alignment_def.contains("extra_mass_penalty")) {
			extra_mass_penalty = p_alignment_def["extra_mass_penalty"].cast<float>();
		}

		return make_matcher(p_query, p_document, p_metric, p_factory,
			std::move(WordRotatorsDistance<Index>(
				WRDOptions{normalize_magnitudes, pyemd, extra_mass_penalty})),
			NoScoreComputer());

	} else {

		std::ostringstream err;
		err << "illegal alignment algorithm " << algorithm;
		throw std::runtime_error(err.str());
	}
}
