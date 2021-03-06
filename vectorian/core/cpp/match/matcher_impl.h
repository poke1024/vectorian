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
	inline float reference_score(
		const Slice &p_slice,
		const float p_matched,
		const float p_unmatched) const {

		// m_matched_weight == 0 indicates that there
		// is no higher relevance of matched content than
		// unmatched content, both are weighted equal (see
		// maximum_internal_score()).

		const float total_score = p_slice.max_sum_of_similarities();

		const float unmatched_weight = std::pow(
			(total_score - p_matched) / total_score,
			m_query->submatch_weight());

		const float reference_score =
			p_matched +
			unmatched_weight * (total_score - p_matched);

		return reference_score;
	}

	template<typename Slice>
	inline float normalized_score(
		const Slice &p_slice,
		const float p_raw_score,
		const std::vector<int16_t> &p_match) const {

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
			p_slice, matched_score, unmatched_score);
	}

	template<typename Slice, typename REVERSE>
	inline MatchRef optimal_match(
		const MatcherRef &matcher,
		const Slice &slice,
		const float p_min_score,
		const REVERSE &reverse) {

		const int len_t = slice.len_t();
		if (len_t <= 0) {
			return m_no_match;
		}

		const int len_s = slice.len_s();
		PPK_ASSERT(len_s > 0);

		m_aligner(m_query, slice, len_s, len_t);

		float raw_score = m_aligner.score();

		float best_final_score = normalized_score(
			slice, raw_score, m_aligner.match());

		if (best_final_score > p_min_score) {

			reverse(m_aligner.mutable_match(), len_s);

			// m_aligner->make_match(matcher, slice, p_min_score)

			return std::make_shared<Match>(
				matcher,
				MatchDigest(m_document, slice.id(), m_aligner.match()),
				best_final_score);
		} else {

			return m_no_match;
		}
	}

public:
	MatcherBase(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const Aligner &p_aligner) :

		Matcher(p_query, p_document, p_metric),
		m_aligner(p_aligner) {

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

template<typename SliceFactory, typename Aligner, bool Bidirectional>
class MatcherImpl : public MatcherBase<Aligner> {

	const SliceFactory m_slice_factory;

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const Aligner &p_aligner,
		const SliceFactory &p_slice_factory) :

		MatcherBase<Aligner>(
			p_query,
			p_document,
			p_metric,
			p_aligner),
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
		const int len_t =  this->m_query->tokens()->size();

		const MatcherRef matcher = this->shared_from_this();

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
				p_matches->worst_score(),
				[] (std::vector<int16_t> &match, int len_s) {});

			if (Bidirectional) {
				const MatchRef m_reverse = this->optimal_match(
					matcher,
					ReversedSlice(slice),
					p_matches->worst_score(),
					reverse_alignment);

				if (m_reverse->score() > m->score()) {
					m = m_reverse;
				}
			}

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
