#include "common.h"
#include "match/matcher.h"
#include "match/match.h"
#include "alignment/aligner.h"
#include "query.h"
#include "document.h"
#include "result_set.h"

template<typename Aligner>
class MatcherBase : public Matcher {
protected:
	Aligner m_aligner;
	MatchRef m_no_match;

	template<typename SCORES, typename REVERSE>
	inline MatchRef optimal_match(
		const int32_t sentence_id,
		const SCORES &scores,
		const int16_t scores_variant_id,
		const float p_min_score,
		const REVERSE &reverse) {

		const int len_s = scores.s_len();
		const int len_t = m_query->len();

		if (len_t < 1 || len_s < 1) {
			return m_no_match;
		}

		m_aligner(scores, len_s, len_t);

		float raw_score = m_aligner.score();

		reverse(m_aligner.mutable_match(), len_s);

		float best_final_score = m_query->normalized_score(
			raw_score, m_aligner.match());

		if (best_final_score > p_min_score) {

			return std::make_shared<Match>(
				this->shared_from_this(),
				scores_variant_id,
				MatchDigest(m_document, sentence_id, m_aligner.match()),
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

		m_aligner.init(
			p_document->max_len_s(),
			m_query->len());
	}

	virtual void initialize() {
		m_no_match = std::make_shared<Match>(
			this->shared_from_this(),
			-1,
			MatchDigest(m_document, -1, std::vector<int16_t>()),
			m_query->min_score()
		);
	}

	virtual float gap_cost(size_t len) const {
		return m_aligner.gap_cost(len);
	}
};

template<typename Slice>
class ReversedSlice {
	const Slice &m_slice;
	const int m_len_s;
	const int m_len_t;

public:
	inline ReversedSlice(const Slice &slice, int len_t) :
		m_slice(slice), m_len_s(slice.s_len()), m_len_t(len_t) {
	}

	inline int s_len() const {
	    return m_len_s;
	}

	inline float score(int u, int v) const {
		return m_slice.score(m_len_s - 1 - u, m_len_t - 1 - v);
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

template<typename Scores, typename Aligner, typename EmbeddingEncoder, bool Bidirectional>
class MatcherImpl : public MatcherBase<Aligner> {

	const std::vector<Scores> m_scores;

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const Aligner &p_aligner,
		const std::vector<Scores> &p_scores) :

		MatcherBase<Aligner>(
			p_query,
			p_document,
			p_metric,
			p_aligner),
		m_scores(p_scores) {

	}

	virtual void match(
		const ResultSetRef &p_matches) {

		std::vector<Scores> good_scores;
		good_scores.reserve(m_scores.size());
		for (const auto &scores : m_scores) {
			if (scores.good()) {
				good_scores.push_back(scores);
			}
		}
		if (good_scores.empty()) {
			return;
		}

		const auto &token_filter = this->m_query->token_filter();

		const auto &slices = this->m_document->sentences();
		const size_t n_slices = slices.size();
		//const size_t max_len_s = m_document->max_len_s();

		size_t token_at = 0;

		for (size_t slice_id = 0;
			slice_id < n_slices && !this->m_query->aborted();
			slice_id++) {

			const auto &slice = slices[slice_id];
			const int len_s = slice.n_tokens;

			if (len_s < 1) {
				continue;
			}

			MatchRef best_slice_match = this->m_no_match;

			for (const auto &scores : good_scores) {

				const auto slice = scores.template create_slice<EmbeddingEncoder>(
				    token_at, len_s, token_filter, EmbeddingEncoder());

				MatchRef m = this->optimal_match(
					slice_id,
					slice,
					scores.variant(),
					p_matches->worst_score(),
					[] (std::vector<int16_t> &match, int len_s) {});

				if (Bidirectional) {
					MatchRef m_reverse = this->optimal_match(
						slice_id,
						ReversedSlice(
                            slice, this->m_query->len()),
						scores.variant(),
						p_matches->worst_score(),
						reverse_alignment);

					if (m_reverse->score() > m->score()) {
						m = m_reverse;
					}
				}

				if (m->score() > best_slice_match->score()) {
					best_slice_match = m;
				}
			}

			if (best_slice_match->score() > this->m_no_match->score()) {

				best_slice_match->compute_scores(
					m_scores.at(best_slice_match->scores_variant_id()), len_s);

				p_matches->add(best_slice_match);
			}

			token_at += len_s;
		}
	}
};
