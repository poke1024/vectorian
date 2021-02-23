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

	template<typename Slice, typename REVERSE>
	inline MatchRef optimal_match(
		const int32_t sentence_id,
		const Slice &slice,
		const int16_t scores_variant_id,
		const float p_min_score,
		const REVERSE &reverse) {

		const int len_s = slice.len_s();
		const int len_t = slice.len_t();

		if (len_t < 1 || len_s < 1) {
			return m_no_match;
		}

		m_aligner(m_query, slice, len_s, len_t);

		float raw_score = m_aligner.score();

		reverse(m_aligner.mutable_match(), len_s);

		float best_final_score = m_query->normalized_score(
			raw_score, m_aligner.match());

		if (best_final_score > p_min_score) {

			/*std::ofstream outfile;
			outfile.open("/Users/arbeit/Desktop/debug_wmd.txt", std::ios_base::app);
			outfile << "final score: " << best_final_score << "\n";
			outfile << "\n";*/

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

void reverse_alignment(std::vector<int16_t> &match, int len_s) {
	for (size_t i = 0; i < match.size(); i++) {
		int16_t u = match[i];
		if (u >= 0) {
			match[i] = len_s - 1 - u;
		}
	}

	std::reverse(match.begin(), match.end());
}

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

		/*std::vector<Scores> good_scores;
		good_scores.reserve(m_scores.size());
		for (const auto &scores : m_scores) {
			if (scores.good()) {
				good_scores.push_back(scores);
			}
		}
		if (good_scores.empty()) {
			return;
		}*/

		const auto &slices = this->m_document->sentences();
		const size_t n_slices = slices.size();
		//const size_t max_len_s = m_document->max_len_s();

		size_t token_at = 0;

		const Token *s_tokens = this->m_document->tokens()->data();
		const Token *t_tokens = this->m_query->tokens()->data();
		const int len_t =  this->m_query->tokens()->size();

		for (size_t slice_id = 0;
			slice_id < n_slices && !this->m_query->aborted();
			slice_id++) {

			const auto &slice_data = slices[slice_id];
			const int len_s = slice_data.n_tokens;

			if (len_s < 1) {
				continue;
			}

			MatchRef best_slice_match = this->m_no_match;

			const auto slice = m_slice_factory.create_slice(
			    TokenSpan{s_tokens + token_at, len_s},
			    TokenSpan{t_tokens, len_t});

			MatchRef m = this->optimal_match(
				slice_id,
				slice,
				0, // variant
				p_matches->worst_score(),
				[] (std::vector<int16_t> &match, int len_s) {});

			if (Bidirectional) {
				const MatchRef m_reverse = this->optimal_match(
					slice_id,
					ReversedSlice(slice),
					0, // variant
					p_matches->worst_score(),
					reverse_alignment);

				if (m_reverse->score() > m->score()) {
					m = m_reverse;
				}
			}

			if (m->score() > best_slice_match->score()) {
				best_slice_match = m;
			}

			if (best_slice_match->score() > this->m_no_match->score()) {

				best_slice_match->compute_scores(
					m_slice_factory, len_s, len_t);

				p_matches->add(best_slice_match);
			}

			token_at += len_s;
		}
	}
};
