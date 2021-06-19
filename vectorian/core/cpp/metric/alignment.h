#ifndef __VECTORIAN_METRIC_ALIGNMENT__
#define __VECTORIAN_METRIC_ALIGNMENT__

#include "common.h"
#include "alignment/aligner.h"
#include "alignment/wmd.h"
#include "alignment/wrd.h"

/*class CosineVSM {
	const QueryRef m_query;

	std::unordered_map<token_t, int32_t> m_t_token_map;
	std::vector<int32_t> m_t_token_count;
	std::vector<int32_t> m_s_token_count;

public:
	VSM(const QueryRef &p_query) : m_query(p_query) {

		const size_t n = m_query->n_tokens();
		const auto &tokens = *m_query->tokens_vector();

		for (size_t i = 0; i < n; i++) {
			const auto id = tokens[i].id;
			const auto it = m_t_token_map.find(id);
			if (it != m_t_token_map.end()) {
				m_t_token_count[it->second] += 1;
			} else {
				const auto k = m_t_token_count.size();
				m_t_token_map[id] = k;
				m_t_token_count.push_back(1);
			}
		}

		m_s_token_count.reserve(m_t_token_count.size());
	}

	template<bool Hook, typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) const {

		const auto shared_vocab_size = m_t_token_count.size();

		m_s_token_count.clear();
		m_s_token_count.resize(shared_vocab_size, 0);

		const size_t len_s = p_slice.len_s();

		for (size_t i = 0; i < len_s; i++) {
			const auto it = m_t_token_map.find(p_slice.s(i).id);
			if (it != m_t_token_map.end()) {
				m_s_token_count[it->second] += 1;
			}
		}

		const float score = 0.0f;

		if (Hook) {
			py::gil_scoped_acquire acquire;
			const auto callback = *p_matcher->query()->debug_hook();
			py::dict data;
			data["score"] = score;
			data["worst_score"] = p_result_set->worst_score();
			callback("alignment/vsm/make", data);
		}

		if (score > p_result_set->worst_score()) {
			auto flow = p_result_set->flow_factory()->create_sparse();
			flow->initialize(shared_vocab_size);

			return p_result_set->add_match(
				p_matcher,
				p_slice.id(),
				flow,
				score);
		} else {
			return MatchRef();
		}
	}
};*/

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

template<typename Aligner>
struct AlignerFactory {
	typedef typename Aligner::IndexType Index;

	const typename Aligner::LocalityType m_locality;
	const typename Aligner::GapCostSpec m_gap_cost_s;
	const typename Aligner::GapCostSpec m_gap_cost_t;

	std::shared_ptr<Aligner> make(
		const size_t max_len_s,
		const size_t max_len_t) const {

		return std::make_shared<Aligner>(
			m_locality,
			m_gap_cost_s,
			m_gap_cost_t,
			max_len_s,
			max_len_t);
	}
};

template<typename Index, typename Aligner>
class InjectiveAlignment {
public:
	typedef AlignerFactory<Aligner> AlignerFactory;

protected:
	const char *m_callback_name;
	const AlignerFactory m_factory;
	std::shared_ptr<Aligner> m_aligner;
	size_t m_max_len_t;
	mutable InjectiveFlowRef<Index> m_cached_flow;

	template<typename Slice>
	void call_debug_hook(
		const QueryRef &p_query,
		const Slice &p_slice,
		const InjectiveFlowRef<Index> &p_flow,
		const float p_score) const {

		py::gil_scoped_acquire acquire;

		py::array_t<float> sim({ p_slice.len_s(), p_slice.len_t() });
		auto mutable_sim = sim.mutable_unchecked<2>();
		for (ssize_t i = 0; i < p_slice.len_s(); i++) {
			for (ssize_t j = 0; j < p_slice.len_t(); j++) {
				mutable_sim(i, j) = p_slice.similarity(i, j);
			}
		}

		py::dict data;
		data["slice"] = p_slice.id();
		data["similarity"] = sim;
		data["values"] = xt::pyarray<float>(m_aligner->matrix(
			p_slice.len_s(), p_slice.len_t()).values_non_neg_ij());
		data["traceback"] = xt::pyarray<float>(m_aligner->matrix(
			p_slice.len_s(), p_slice.len_t()).traceback());
		data["flow"] = p_flow->to_py();
		data["score"] = p_score;

		const auto callback = *p_query->debug_hook();
		callback(m_callback_name, data);
	}

public:
	InjectiveAlignment(
		const char *p_callback_name,
		const AlignerFactory &p_factory) :

		m_callback_name(p_callback_name),
		m_factory(p_factory),
		m_max_len_t(0) {
	}

	void init(const size_t max_len_s, const size_t max_len_t) {
		m_aligner = m_factory.make(max_len_s, max_len_t);
		m_max_len_t = max_len_t;
	}

	template<bool Hook, typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) const {

		if (!m_cached_flow) {
			m_cached_flow = p_result_set->flow_factory()->create_injective();
			m_cached_flow->reserve(m_max_len_t);
		}

		const float aligner_score = m_aligner->compute(
			*m_cached_flow.get(),
			[&p_slice] (auto i, auto j) -> float {
				return p_slice.similarity(i, j);
			},
			p_slice.len_s(),
			p_slice.len_t());

		const float score_max = reference_score(
			p_matcher->query(), p_slice, m_cached_flow->max_score(p_slice));
		const auto score = Score(aligner_score, score_max);

		if (Hook) {
			call_debug_hook(
				p_matcher->query(), p_slice, m_cached_flow, aligner_score);
		}

		if (score > p_result_set->worst_score()) {
			const auto flow = m_cached_flow;
			m_cached_flow.reset();

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

			const Token *s_tokens = p_match->document()->tokens_vector()->data();
			const auto &t_tokens = p_match->query()->tokens_vector();

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

template<typename Index, typename Locality>
class AlignmentWithAffineGapCost : public InjectiveAlignment<
	Index, alignments::AffineGapCostAligner<Locality, Index, float>> {
public:
	typedef alignments::AffineGapCostAligner<Locality, Index, float> Aligner;

	AlignmentWithAffineGapCost(
		const float p_gap_cost_s,
		const float p_gap_cost_t,
		const Locality &p_locality) :

		InjectiveAlignment<Index, Aligner>(
			"alignment/affine-gap-cost",
			AlignerFactory<Aligner>{p_locality, p_gap_cost_s, p_gap_cost_t}) {
	}

	inline float gap_cost_s(size_t len) const {
		return this->m_aligner->gap_cost_s(len);
	}

	inline float gap_cost_t(size_t len) const {
		return this->m_aligner->gap_cost_t(len);
	}
};

template<typename Index, typename Locality>
class AlignmentWithGeneralGapCost : public InjectiveAlignment<
	Index, alignments::GeneralGapCostAligner<Locality, Index, float>> {
public:
	typedef alignments::GeneralGapCostAligner<Locality, Index, float> Aligner;

	AlignmentWithGeneralGapCost(
		const alignments::GapTensorFactory &p_gap_cost_s,
		const alignments::GapTensorFactory &p_gap_cost_t,
		const Locality &p_locality) :

		InjectiveAlignment<Index, alignments::GeneralGapCostAligner<Locality, Index, float>>(
			"alignment/general-gap-cost",
			AlignerFactory<Aligner>{p_locality, p_gap_cost_s, p_gap_cost_t}) {
	}

	inline float gap_cost_s(size_t len) const {
		return this->m_aligner->gap_cost_s(len);
	}

	inline float gap_cost_t(size_t len) const {
		return this->m_aligner->gap_cost_t(len);
	}
};

class NoScoreComputer {
public:
	inline void operator()(const MatchRef &) const {
		// no op.
	}
};

template<typename Index>
class WordMoversDistance {
	const WMDOptions m_options;

	BOWBuilder<Index, UntaggedTokenFactory> m_untagged_builder;
	BOWBuilder<Index, TaggedTokenFactory> m_tagged_builder;
	UniqueTokensBOWBuilder<Index> m_unique_builder;

	WMD<Index> m_wmd;

	template<typename Slice, typename Solver>
	inline WMDSolution<typename Solver::FlowRef> compute(
		const QueryRef &p_query,
		const Slice &p_slice,
		const Solver &p_solver) {

		switch (p_slice.similarity_dependency()) {
			case NONE: {
				return m_wmd(
					p_query, p_slice,
					m_untagged_builder,
					m_options, p_solver);
			} break;

			case TAGS: {
				return m_wmd(
					p_query, p_slice,
					m_tagged_builder,
					m_options, p_solver);
			} break;

			case POSITION: {
				return m_wmd(
					p_query, p_slice,
					m_unique_builder,
					m_options, p_solver);
			};

			default: {
				throw std::runtime_error("unsupported similarity dependency in WMD");
			} break;
		}
	}

	template<bool Hook, typename Slice, typename Solver>
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

		const float score_max = reference_score(
			p_matcher->query(), p_slice, r.flow->max_score(p_slice));
		const auto score = Score(r.score, score_max);

		if (Hook) {
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

		m_options(p_options),
		m_untagged_builder(UntaggedTokenFactory()),
		m_tagged_builder(TaggedTokenFactory()) {
	}

	void init(Index max_len_s, Index max_len_t) {
		m_untagged_builder.allocate(max_len_s, max_len_t);
		m_tagged_builder.allocate(max_len_s, max_len_t);

		m_wmd.allocate(max_len_s, max_len_t);
	}

	inline float gap_cost_s(size_t len) const {
		return 0;
	}

	inline float gap_cost_t(size_t len) const {
		return 0;
	}

	template<bool Hook, typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) {

		const FlowFactoryRef<Index> flow_factory =
			p_result_set->flow_factory();

		if (m_options.relaxed) {
			return make_match<Hook>(
				p_matcher,
				p_slice,
				p_result_set,
				typename AbstractWMD<Index>::RelaxedSolver(flow_factory));
		} else {
			return make_match<Hook>(
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

	inline float gap_cost_s(size_t len) const {
		return 0;
	}

	inline float gap_cost_t(size_t len) const {
		return 0;
	}

	template<bool Hook, typename Slice>
	inline MatchRef make_match(
		const MatcherRef &p_matcher,
		const Slice &p_slice,
		const ResultSetRef &p_result_set) {

		const FlowFactoryRef<Index> flow_factory =
			p_result_set->flow_factory();

		const auto r = m_wrd.compute(
			p_matcher->query(), p_slice, flow_factory, m_options);

		const float score_max = reference_score(
			p_matcher->query(), p_slice, r.flow->max_score(p_slice));
		const auto score = Score(r.score, score_max);

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
	const py::dict &alignment_def) {

	if (alignment_def.contains("algorithm")) {
		return alignment_def["algorithm"].cast<py::str>();
	} else {
		return "alignment/local"; // default
	}
}

inline MatcherOptions create_alignment_matcher_options(
	const py::dict &alignment_def) {

	const std::string algorithm = get_alignment_algorithm(alignment_def);
	return MatcherOptions{algorithm == "word-rotators-distance", alignment_def};
}

inline GapMask parse_gap_mask(const py::dict &alignment_def) {
	GapMask gap{true, true};

	if (alignment_def.contains("gap_mask")) {
		const auto m = alignment_def["gap_mask"].cast<std::string>();
		gap.u = m.find("s") != std::string::npos; // document
		gap.v = m.find("t") != std::string::npos; // query
	}

	return gap;
}

template<
	template<typename, typename> typename Alignment,
	typename Index, typename SliceFactory, typename Locality, typename GapCost>
MatcherRef make_alignment_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const SliceFactory &p_factory,
	const Locality &p_locality,
	const GapCost &p_gap_cost_s,
	const GapCost &p_gap_cost_t) {

	return make_matcher(p_query, p_document, p_metric, p_factory,
		std::move(Alignment<Index, Locality>(p_gap_cost_s, p_gap_cost_t, p_locality)),
		Alignment<Index, Locality>::create_score_computer(p_factory));
}

inline std::vector<std::string> split_str(const std::string &s, const char sep) {
	std::istringstream is(s);
	std::string token;
	std::vector<std::string> parts;
	while (std::getline(is, token, sep)) {
		parts.push_back(token);
	}
	return parts;
}

template<typename Index, typename SliceFactory>
MatcherRef create_alignment_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const MatcherOptions &p_matcher_options,
	const SliceFactory &p_factory) {

	const py::dict &alignment_def = p_matcher_options.alignment_def;

	const std::string algorithm = get_alignment_algorithm(alignment_def);
	std::vector<std::string> algo_parts = split_str(algorithm, '/');

	/*if (algorithm == "cosine-vector-space-model") {

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(CosineVSM<Index>(p_query)),
			CosineVSM<Index>::create_score_computer(p_factory));

	} else*/

	if (algo_parts.size() == 3 && algo_parts[0] == "alignment") {

		if (algo_parts[2] == "general") { // general gap costs

			PPK_ASSERT(alignment_def.contains("gap"));
			const auto costs_dict = alignment_def["gap"].cast<py::dict>();

			const auto gap_cost_s = costs_dict["s"].cast<alignments::GapTensorFactory>();
			const auto gap_cost_t = costs_dict["t"].cast<alignments::GapTensorFactory>();

			if (algo_parts[1] == "local") {
				float zero = 0.5;
				if (alignment_def.contains("zero")) {
					zero = alignment_def["zero"].cast<float>();
				}

				return make_alignment_matcher<AlignmentWithGeneralGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::Local<float>(zero), gap_cost_s, gap_cost_t);
			} else if (algo_parts[1] == "global") {
				return make_alignment_matcher<AlignmentWithGeneralGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::Global<float>(), gap_cost_s, gap_cost_t);
			} else if (algo_parts[1] == "semiglobal") {
				return make_alignment_matcher<AlignmentWithGeneralGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::SemiGlobal<float>(), gap_cost_s, gap_cost_t);
			} else {
				throw std::invalid_argument(algo_parts[2]);
			}
		}

		else if (algo_parts[2] == "affine") { // affine gap costs

			PPK_ASSERT (alignment_def.contains("gap"));
			const auto costs_dict = alignment_def["gap"].cast<py::dict>();
			const float gap_cost_s = costs_dict["s"].cast<float>();
			const float gap_cost_t = costs_dict["t"].cast<float>();

			if (algo_parts[1] == "local") {
				float zero = 0.5f;
				if (alignment_def.contains("zero")) {
					zero = alignment_def["zero"].cast<float>();
				}

				return make_alignment_matcher<AlignmentWithAffineGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::Local<float>(zero), gap_cost_s, gap_cost_t);
			} else if (algo_parts[1] == "global") {
				return make_alignment_matcher<AlignmentWithAffineGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::Global<float>(), gap_cost_s, gap_cost_t);
			} else if (algo_parts[1] == "semiglobal") {
				return make_alignment_matcher<AlignmentWithAffineGapCost, Index, SliceFactory>(
					p_query, p_document, p_metric, p_factory,
					alignments::SemiGlobal<float>(), gap_cost_s, gap_cost_t);
			} else {
				throw std::invalid_argument(algo_parts[2]);
			}
		}

		else {
			throw std::invalid_argument(algo_parts[1]);
		}

	} else if (algorithm == "word-movers-distance") {

		bool relaxed = true;
		bool normalize_bow = true;
		bool symmetric = true;
		bool injective = true;
		float extra_mass_penalty = -1.0f;

		if (alignment_def.contains("relaxed")) {
			relaxed = alignment_def["relaxed"].cast<bool>();
		}
		if (alignment_def.contains("normalize_bow")) {
			normalize_bow = alignment_def["normalize_bow"].cast<bool>();
		}
		if (alignment_def.contains("symmetric")) {
			symmetric = alignment_def["symmetric"].cast<bool>();
		}
		if (alignment_def.contains("injective")) {
			injective = alignment_def["injective"].cast<bool>();
		}
		if (alignment_def.contains("extra_mass_penalty")) {
			extra_mass_penalty = alignment_def["extra_mass_penalty"].cast<float>();
		}

		return make_matcher(
			p_query, p_document, p_metric, p_factory,
			std::move(WordMoversDistance<Index>(WMDOptions{
				relaxed, normalize_bow, symmetric, injective, pyemd, extra_mass_penalty})),
			NoScoreComputer());

	} else if (algorithm == "word-rotators-distance") {

		PPK_ASSERT(p_matcher_options.needs_magnitudes);

		bool normalize_magnitudes = true;
		if (alignment_def.contains("normalize_magnitudes")) {
			normalize_magnitudes = alignment_def["normalize_magnitudes"].cast<bool>();
		}

		float extra_mass_penalty = -1.0f;
		if (alignment_def.contains("extra_mass_penalty")) {
			extra_mass_penalty = alignment_def["extra_mass_penalty"].cast<float>();
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

#endif // __VECTORIAN_METRIC_ALIGNMENT__
