#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"

class StaticEmbeddingMetric : public Metric {
protected:
	const EmbeddingRef m_embedding;
	const py::dict m_options;
	const py::dict m_alignment_def;

	MatrixXf m_similarity;
	ArrayXf m_magnitudes[2];

	void compute_magnitudes(
		const WordVectors &p_embeddings,
		const VocabularyToEmbedding &p_vocabulary_to_embedding,
		const Needle &p_needle) {

		/*r_mag_s.resize(p_vocabulary_to_embedding.size());
		p_vocabulary_to_embedding.iterate([&] (const auto &x, const size_t offset) {
			const auto n = x.rows();
			for (size_t i = 0; i < n; i++) {
				r_mag_s(offset + i) = m_embeddings.unmodified[x(i)].norm();
			}
		});

		r_mag_t.resize(p_needle.size());
		for (size_t j = 0; j < p_needle.size(); j++) {
			const size_t k = p_needle.embedding_token_ids()[j];
			r_mag_t(j) = m_embeddings.unmodified[k].norm();
		}*/
	}

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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

		//compute_length();

		if (p_sent_metric_def.contains("similarity_falloff")) {
			const float similarity_falloff = p_sent_metric_def["similarity_falloff"].cast<float>();
			m_similarity = m_similarity.array().pow(similarity_falloff);
		}

	}

	inline const py::dict &options() const {
		return m_options;
	}

	inline const MatrixXf &similarity() const {
		return m_similarity;
	}

	inline float magnitude_s(int i) const {
		return 0.0f; // FIXME
	}

	inline float magnitude_t(int i) const {
		return 0.0f; // FIXME
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
