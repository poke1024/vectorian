#include "common.h"
#include "match/matcher.h"
#include "match/match.h"
#include "alignment/aligner.h"
#include "query.h"
#include "document.h"
#include "result_set.h"
#include <fstream>

template<typename Aligner>
class MatcherBase : public Matcher {
protected:
	Aligner m_aligner;
	MatchRef m_no_match;

	template<typename Slice>
	inline MatchRef optimal_match(
		const MatcherRef &matcher,
		const Slice &slice,
		const float p_min_score) {

		const MatchRef m = m_aligner.make_match(
			matcher, slice, p_min_score);

		if (m.get()) {
			return m;
		} else {
			return m_no_match;
		}
	}

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
			m_query->len());
	}

	virtual void initialize() {
		m_no_match = std::make_shared<Match>(
			this->shared_from_this(),
			MatchDigest(m_document, -1, std::vector<int16_t>()),
			m_query->min_score()
		);
	}

	virtual float gap_cost(size_t len) const {
		return m_aligner.gap_cost(len);
	}
};

void reverse_alignment(std::vector<int16_t> &match, int len_s) {
	for (size_t i = 0; i < match.size(); i++) {
		int16_t u = match[i];
		if (u >= 0) {
			match[i] = len_s - 1 - u;
		}
	}

	std::reverse(match.begin(), match.end());
}

/*template<typename S>
inline size_t compute_len_s(
	const S &p_slices,
	const size_t p_token_at,
	const size_t p_index,
	const size_t p_window_size) {

	const size_t n_slices = slices.size();

	size_t len_s = 0;
	const auto &slice_data = slices[p_index + p_window_size];
	return slice_data.token_at - p_token_at;
}*/

template<typename SliceFactory, typename Aligner>
class MatcherImpl : public MatcherBase<Aligner> {

	const SliceFactory m_slice_factory;

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		Aligner &&p_aligner,
		const SliceFactory &p_slice_factory) :

		MatcherBase<Aligner>(
			p_query,
			p_document,
			p_metric,
			std::move(p_aligner)),
		m_slice_factory(p_slice_factory) {
	}

	virtual void match(
		const ResultSetRef &p_matches) {

		const auto &slice_strategy = this->m_query->slice_strategy();

		const auto spans = this->m_document->spans(slice_strategy.level);
		const size_t n_slices = spans->size();
		//const size_t max_len_s = m_document->max_len_s();

		size_t token_at = 0;

		const Token *s_tokens = this->m_document->tokens()->data();
		const Token *t_tokens = this->m_query->tokens()->data();
		const int len_t = this->m_query->tokens()->size();
		if (len_t < 1) {
			return; // no matches
		}

		const MatcherRef matcher = this->shared_from_this();
		//this->needs_magnitudes()

		for (size_t slice_id = 0;
			slice_id < n_slices && !this->m_query->aborted();
			slice_id += slice_strategy.window_step) {

			const int len_s = spans->safe_len(
				slice_id, slice_strategy.window_size);

			if (len_s < 1) {
				continue;
			}

			const auto slice = m_slice_factory.create_slice(
				slice_id,
			    TokenSpan{s_tokens + token_at, len_s},
			    TokenSpan{t_tokens, len_t});

			MatchRef m = this->optimal_match(
				matcher,
				slice,
				p_matches->worst_score());

			if (m->score() > this->m_no_match->score()) {
				m->compute_scores(
					m_slice_factory, len_s, len_t);

				p_matches->add(m);
			}

			token_at += spans->safe_len(
				slice_id, slice_strategy.window_step);
		}
	}
};
