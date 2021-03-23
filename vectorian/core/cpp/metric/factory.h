#include "match/matcher_impl.h"

template<typename SliceFactory, typename Aligner, typename Finalizer>
MatcherRef make_matcher(
	const QueryRef &p_query,
	const DocumentRef &p_document,
	const MetricRef &p_metric,
	const SliceFactory &p_factory,
	Aligner &&p_aligner,
	const Finalizer &p_finalizer) {

	return std::make_shared<MatcherImpl<SliceFactory, Aligner, Finalizer>>(
		p_query, p_document, p_metric, std::move(p_aligner), p_finalizer, p_factory);
}

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

		if (token_filter.all()) {
			return m_make_matcher(SliceFactory(m_make_slice));
		} else {
			return m_make_matcher(FilteredSliceFactory(
				p_query,
				SliceFactory(m_make_slice),
				p_document, token_filter));
		}
	}
};