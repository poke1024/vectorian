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
