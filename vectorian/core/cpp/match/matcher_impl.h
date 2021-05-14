#ifndef __VECTORIAN_MATCHER_IMPL__
#define __VECTORIAN_MATCHER_IMPL__

#include "common.h"
#include "match/matcher.h"
#include "match/match.h"
#include "alignment/aligner.h"
#include "query.h"
#include "document.h"
#include "result_set.h"
#include "metric/alignment.h"
#include <fstream>

template<typename Aligner>
class MatcherBase : public Matcher {
protected:
	Aligner m_aligner;
	MatchRef m_no_match;

public:
	MatcherBase(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		Aligner &&p_aligner) :

		Matcher(p_query, p_document, p_metric),
		m_aligner(std::move(p_aligner)) {

		const auto &slice_strategy = p_query->slice_strategy();

		m_aligner.init(
			p_document->spans(slice_strategy.level)->max_len(slice_strategy.window_size),
			m_query->n_tokens());
	}

	virtual void initialize() {
		m_no_match = std::make_shared<Match>(
			this->shared_from_this(),
			MatchDigest(m_document, -1, FlowRef<int16_t>()),
			Score(m_query->min_score(), 1)
		);
	}

	virtual float gap_cost(size_t len) const {
		return m_aligner.gap_cost(len);
	}

	virtual GapMask gap_mask() const {
		return m_aligner.gap_mask();
	}
};

inline void reverse_alignment(std::vector<int16_t> &match, int len_s) {
	for (size_t i = 0; i < match.size(); i++) {
		int16_t u = match[i];
		if (u >= 0) {
			match[i] = len_s - 1 - u;
		}
	}

	std::reverse(match.begin(), match.end());
}

template<typename SliceFactory, typename Aligner, typename Finalizer>
class MatcherImpl : public MatcherBase<Aligner> {
	const Finalizer m_finalizer;
	const SliceFactory m_slice_factory;

	template<bool Hook, typename RunMatch>
	void run_matches(
		const ResultSetRef &p_matches,
		const RunMatch &p_run_match) {

		const auto &slice_strategy = this->m_query->slice_strategy();

		const Token *s_tokens = this->m_document->tokens_vector()->data();
		const Token *t_tokens = this->m_query->tokens_vector()->data();
		const auto len_t = this->m_query->n_tokens();
		if (len_t < 1) {
			return; // no matches
		}

		const MatcherRef matcher = this->shared_from_this();
		const auto spans = this->m_document->spans(slice_strategy.level);

		const auto match_span = [&, s_tokens, t_tokens, len_t] (
			const size_t slice_id, const size_t token_at, const size_t len_s) {

			p_run_match([&, s_tokens, t_tokens, slice_id, token_at, len_s, len_t] () {

				const auto slice = m_slice_factory.create_slice(
					slice_id,
				    TokenSpan{s_tokens, static_cast<int32_t>(token_at), static_cast<int32_t>(len_s)},
				    TokenSpan{t_tokens, 0, static_cast<int32_t>(len_t)});

				return this->m_aligner.template make_match<Hook>(matcher, slice, p_matches);
			});

			return !this->m_query->aborted();
		};

		spans->iterate(slice_strategy, match_span);
	}

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		Aligner &&p_aligner,
		const Finalizer &p_finalizer,
		const SliceFactory &p_slice_factory) :

		MatcherBase<Aligner>(
			p_query,
			p_document,
			p_metric,
			std::move(p_aligner)),
		m_finalizer(p_finalizer),
		m_slice_factory(p_slice_factory) {
	}

	virtual void match(
		const ResultSetRef &p_matches) {

		PPK_ASSERT(p_matches->size() == 0);

		if (this->m_query->debug_hook().has_value()) {

			run_matches<true>(p_matches, [this] (const auto &f) -> MatchRef {
				const std::chrono::steady_clock::time_point time_begin =
					std::chrono::steady_clock::now();

				const auto m = f();

				{
					py::gil_scoped_acquire acquire;
					const std::chrono::steady_clock::time_point time_end =
						std::chrono::steady_clock::now();
					const auto delta_time = std::chrono::duration_cast<std::chrono::microseconds>(
						time_end - time_begin).count();
					const auto callback = *this->m_query->debug_hook();
					callback("document/match_time", delta_time);
				}

				return m;
			});

		} else {

			run_matches<false>(p_matches, [] (const auto &f) -> MatchRef {
				return f();
			});
		}

		if (this->m_query->debug_hook().has_value()) {
			py::gil_scoped_acquire acquire;
			py::dict data;
			data["doc_id"] = this->m_document->id();
			data["num_results"] = p_matches->size();
			const auto callback = *this->m_query->debug_hook();
			callback("document/done", data);
		}

		p_matches->modify([this] (const auto &match) {
			this->m_finalizer(match);
		});
	}
};

template<typename MakeSlice, typename MakeMatcher>
class FilteredMatcherFactory {
	const MakeSlice m_make_slice;
	const MakeMatcher m_make_matcher;

public:
	typedef typename std::invoke_result<
		MakeSlice,
		const size_t,
		const TokenSpan&,
		const TokenSpan&>::type Slice;

	FilteredMatcherFactory(
		const MakeSlice &make_slice,
		const MakeMatcher &make_matcher) :

		m_make_slice(make_slice),
		m_make_matcher(make_matcher) {
	}

	MatcherRef create(
		const QueryRef &p_query,
		const DocumentRef &p_document) const {

		const auto token_filter = p_query->token_filter();

		if (token_filter.get()) {
			return m_make_matcher(FilteredSliceFactory(
				p_query,
				SliceFactory(m_make_slice),
				p_document, token_filter));
		} else {
			return m_make_matcher(SliceFactory(m_make_slice));
		}
	}
};

template<typename GenSlice>
MatcherRef MinimalMatcherFactory::make_matcher(
	const QueryRef &p_query,
	const MetricRef &p_metric,
	const DocumentRef &p_document,
	const MatcherOptions &p_matcher_options,
	const GenSlice &p_gen_slice) const {

	const auto gen_matcher = [p_query, p_document, p_metric, p_matcher_options] (auto slice_factory) {
		return create_alignment_matcher<Index>(
			p_query, p_document, p_metric, p_matcher_options, slice_factory);
	};

	FilteredMatcherFactory factory(
		p_gen_slice,
		gen_matcher);

	return factory.create(p_query, p_document);
}

#endif // __VECTORIAN_MATCHER_IMPL__
