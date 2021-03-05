#include "embedding/static.h"
#include "metric/static.h"
#include "query.h"

Needle::Needle(
	const QueryRef &p_query,
	const VocabularyToEmbedding &p_vocabulary_to_embedding) :

	m_needle(p_query->tokens()) {

	const auto &needle = *m_needle;

	m_needle_vocabulary_token_ids.resize({needle.size()});
	for (size_t i = 0; i < needle.size(); i++) {
		m_needle_vocabulary_token_ids(i) = needle[i].id;
	}

	// p_a maps from a Vocabulary corpus token id to an Embedding token id,
	// e.g. 3 in the corpus and 127 in the embedding.

	// p_b are the needle's Vocabulary token ids (not yet mapped to Embedding)

	m_needle_embedding_token_ids.resize({needle.size()});

	for (size_t i = 0; i < needle.size(); i++) {
		const token_t t = m_needle_vocabulary_token_ids(i);
		if (t >= 0) {
			token_t mapped = -1;
			token_t r = t;
			for (const auto &x : p_vocabulary_to_embedding.unpack()) {
				if (r < static_cast<ssize_t>(x.shape(0))) {
					mapped = x[r];
					break;
				} else {
					r -= x.shape(0);
				}
			}
			PPK_ASSERT(mapped >= 0);
			m_needle_embedding_token_ids(i) = mapped; // map to Embedding token ids
		} else {
			m_needle_embedding_token_ids(i) = -1;
		}
	}
}

MetricRef StaticEmbedding::create_metric(
	const QueryRef &p_query,
	const WordMetricDef &p_metric,
	const py::dict &p_sent_metric_def,
	const VocabularyToEmbedding &p_vocabulary_to_embedding) {

	return std::make_shared<StaticEmbeddingMetric>(
		p_query,
		std::dynamic_pointer_cast<StaticEmbedding>(shared_from_this()),
		p_metric,
		p_sent_metric_def,
		p_vocabulary_to_embedding);
}
