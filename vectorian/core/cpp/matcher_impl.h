#include "matcher.h"
#include "mismatch_penalty.h"
#include "aligner.h"
#include "match.h"
#include "query.h"
#include "document.h"
#include "result_set.h"

const float smith_waterman_zero = 0.5;

class MatcherBase : public Matcher {
protected:
	const QueryRef m_query;
	const DocumentRef m_document;
	const MetricRef m_metric;

	const MismatchPenaltyRef m_mismatch_penalty;
	const MatchRef m_no_match;

	Aligner<int16_t, float> m_aligner;

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

		const auto &gap_cost = *m_mismatch_penalty.get();

		m_aligner.waterman_smith_beyer(
			scores,
			gap_cost,
			len_s,
			len_t,
			smith_waterman_zero);

		float raw_score = m_aligner.score();

		reverse(m_aligner.mutable_match(), len_s);

		float best_final_score = m_query->normalized_score(
			raw_score, m_aligner.match());

		if (best_final_score > p_min_score) {

			return std::make_shared<Match>(
				m_query,
				m_metric,
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
		const MismatchPenaltyRef &p_mismatch_penalty) :

		m_query(p_query),
		m_document(p_document),
		m_metric(p_metric),
		m_mismatch_penalty(p_mismatch_penalty),
		m_no_match(std::make_shared<Match>(
			m_query,
			m_metric,
			-1,
			MatchDigest(m_document, -1, std::vector<int16_t>()),
			p_query->min_score()
		)),
		m_aligner(p_document->max_len_s(), m_query->len()) {
	}

};

template<typename Scores>
class ReversedScores {
	const Scores &m_scores;
	const int m_len_s;
	const int m_len_t;

public:
	inline ReversedScores(const Scores &scores, int len_t) :
		m_scores(scores), m_len_s(scores.s_len()), m_len_t(len_t) {
	}

	inline int s_len() const {
	    return m_len_s;
	}

	inline float operator()(int u, int v) const {
		return m_scores(m_len_s - 1 - u, m_len_t - 1 - v);
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

template<typename Scores>
class MatcherImpl : public MatcherBase {
	const std::vector<Scores> m_scores;

public:
	MatcherImpl(
		const QueryRef &p_query,
		const DocumentRef &p_document,
		const MetricRef &p_metric,
		const std::vector<Scores> &p_scores) :

		MatcherBase(
			p_query,
			p_document,
			p_metric,
			std::make_shared<MismatchPenalty>(
				p_query->mismatch_length_penalty(),
				p_document->max_len_s())
		),
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

		const int pos_filter = m_query->ignore_determiners() ?
		    m_document->vocabulary()->det_pos() : -1;

		const auto &sentences = m_document->sentences();
		const size_t n_sentences = sentences.size();
		//const size_t max_len_s = m_document->max_len_s();

		size_t token_at = 0;

		for (size_t sentence_id = 0;
			sentence_id < n_sentences && !m_query->aborted();
			sentence_id++) {

			const Sentence &sentence = sentences[sentence_id];
			const int len_s = sentence.n_tokens;

			if (len_s < 1) {
				continue;
			}

			MatchRef best_sentence_match = m_no_match;

			for (const auto &scores : good_scores) {

					const auto sentence_scores = scores.create_sentence_scores(
					    token_at, len_s, pos_filter);

					MatchRef m = optimal_match(
						sentence_id,
						sentence_scores,
						scores.variant(),
						p_matches->worst_score(),
						[] (std::vector<int16_t> &match, int len_s) {});

					if (m_query->bidirectional()) {
						MatchRef m_reverse = optimal_match(
							sentence_id,
							ReversedScores(
    							sentence_scores, m_query->len()),
							scores.variant(),
							p_matches->worst_score(),
							reverse_alignment);

						if (m_reverse->score() > m->score()) {
							m = m_reverse;
						}
					}

#if 0
					if (m_document->id() == 17 && sentence_id == 287) {
						m_matrix.print(m_query, m_document, sentence_id, m_query->len(), len_s);
						printf("match score: %.1f\n", 100.0f * m->score());

						int kk = 0;
						for (auto m : m_matrix.match) {
							if (m >= 0) {
								printf("m %d / %d\n", m, kk);
								printf("sim: %f\n", sentence_scores.similarity(m, kk));
								printf("weight: %f\n", sentence_scores.weight(m, kk));
								printf("combined: %f\n", sentence_scores(m, kk));
							}
							kk++;
						}


					}
#endif

					if (m->score() > best_sentence_match->score()) {
						best_sentence_match = m;
					}
			}

			if (best_sentence_match->score() > m_no_match->score()) {

				best_sentence_match->compute_scores(
					m_scores.at(best_sentence_match->scores_variant_id()), len_s);

#if 0
				if (m_document->id() == 17 && sentence_id == 287) {
					 best_sentence_match->print_scores();
				}
#endif

				p_matches->add(best_sentence_match);
			}

			token_at += len_s;
		}
	}
};
