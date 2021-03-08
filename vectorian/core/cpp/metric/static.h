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
		const QueryRef &p_query,
		const StaticEmbeddingRef &p_embedding,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const VocabularyToEmbedding &p_vocabulary_to_embedding);

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

	virtual MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

	virtual const std::string &name() const;

	inline const py::dict &alignment_def() const {
		return m_alignment_def;
	}
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
