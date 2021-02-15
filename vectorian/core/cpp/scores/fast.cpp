#include "scores/fast.h"
#include "document.h"
#include "query.h"

FastScores::FastScores(
    const QueryRef &p_query,
    const DocumentRef &p_document,
    const FastMetricRef &p_metric) :

    m_query(p_query),
    m_document(p_document),
    m_metric(p_metric) {

    m_filtered.resize(p_document->max_len_s());
}

FastSentenceScores FastScores::create_sentence_scores(
	const size_t p_s_offset,
	const size_t p_s_len,
	const int p_pos_filter) const {

	const Token *s_tokens = m_document->tokens()->data();
	const Token *t_tokens = m_query->tokens()->data();

	if (p_pos_filter > -1) {
	    const Token *s = s_tokens + p_s_offset;
	    Token *new_s = m_filtered.data();
        PPK_ASSERT(p_s_len <= m_filtered.size());

	    size_t new_s_len = 0;
        for (size_t i = 0; i < p_s_len; i++) {
            if (s[i].pos != p_pos_filter) {
                new_s[new_s_len++] = s[i];
            }
        }

        return FastSentenceScores(
            m_metric,
            new_s,
            new_s_len,
            t_tokens);
	}
    else {
        return FastSentenceScores(
            m_metric,
            s_tokens + p_s_offset,
            p_s_len,
            t_tokens);
    }
}
