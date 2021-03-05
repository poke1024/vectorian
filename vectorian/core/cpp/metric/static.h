#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"

class StaticEmbeddingMetric : public Metric {
protected:
	const EmbeddingRef m_embedding;
	const py::dict m_options;
	const py::dict m_alignment_def;

	xt::xtensor<float, 2> m_similarity;
	xt::xtensor<float, 1> m_mag_s;
	xt::xtensor<float, 1> m_mag_t;

	void compute_magnitudes(
		const WordVectors &p_embeddings,
		const VocabularyToEmbedding &p_vocabulary_to_embedding,
		const Needle &p_needle) {

		m_mag_s.resize({p_vocabulary_to_embedding.size()});
		p_vocabulary_to_embedding.iterate([&] (const auto &x, const size_t offset) {
			const size_t n = static_cast<size_t>(x.shape(0));
			PPK_ASSERT(offset + n <= static_cast<size_t>(m_mag_s.shape(0)));
			for (size_t i = 0; i < n; i++) {
				const token_t k = x(i);
				if (k >= 0) {
					const auto row = xt::view(p_embeddings.unmodified, k, xt::all());
					m_mag_s(offset + i) = xt::linalg::norm(row);
				} else {
					m_mag_s(offset + i) = 0.0f;
				}
			}
		});

		m_mag_t.resize({p_needle.size()});
		for (size_t j = 0; j < p_needle.size(); j++) {
			const token_t k = p_needle.embedding_token_ids()[j];
			if (k >= 0) {
				const auto row = xt::view(p_embeddings.unmodified, k, xt::all());
				m_mag_t(j) = xt::linalg::norm(row);
			} else {
				m_mag_t(j) = 0.0f;
			}
		}
	}

public:
	StaticEmbeddingMetric(const py::dict &p_sent_metric_def) :
		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()){
	}

	StaticEmbeddingMetric(
		const StaticEmbeddingRef &p_embedding,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const VocabularyToEmbedding &p_vocabulary_to_embedding,
		const std::vector<Token> &p_needle) :

		m_embedding(p_embedding),
		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()) {

		const auto builder = p_metric.instantiate(
			p_embedding->embeddings());

		const Needle needle(p_vocabulary_to_embedding, p_needle);

		builder->build_similarity_matrix(
			p_vocabulary_to_embedding,
			needle,
			m_similarity);

		if (p_sent_metric_def.contains("similarity_falloff")) {
			const float similarity_falloff = p_sent_metric_def["similarity_falloff"].cast<float>();
			m_similarity = xt::pow(m_similarity, similarity_falloff);
		}

		// FIXME do not do this always.
		compute_magnitudes(
			p_embedding->embeddings(),
			p_vocabulary_to_embedding,
			needle);
	}

	inline const py::dict &options() const {
		return m_options;
	}

	inline const xt::xtensor<float, 2> &similarity() const {
		return m_similarity;
	}

	inline float magnitude_s(int i) const {
		return m_mag_s(i);
	}

	inline float magnitude_t(int i) const {
		return m_mag_t(i);
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const;

	inline const py::dict &alignment_def() const {
		return m_alignment_def;
	}
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
