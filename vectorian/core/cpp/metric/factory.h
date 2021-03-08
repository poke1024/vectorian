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

template<typename MakeSlice>
class FactoryGenerator {
	const MakeSlice m_make_slice;

public:
	typedef typename std::invoke_result<
		MakeSlice,
		const size_t,
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
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const TokenFilter &p_token_filter) const {

		return FilteredSliceFactory(
			p_query,
			create(p_document),
			p_document, p_token_filter);
	}
};