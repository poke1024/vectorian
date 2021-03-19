#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"

class StaticEmbeddingMetric : public Metric {
protected:
	const std::vector<StaticEmbeddingRef> m_embeddings;
	const py::dict m_options;
	const py::dict m_alignment_def;

	bool m_needs_magnitudes;
	MatcherFactoryRef m_matcher_factory;

	xt::xtensor<float, 2> m_similarity;
	xt::xtensor<float, 1> m_mag_s;
	xt::xtensor<float, 1> m_mag_t;

	void compute_magnitudes(
		const QueryVocabularyRef &p_vocabulary,
		const Needle &p_needle) {

		m_mag_s.resize({p_vocabulary->size()});
		size_t offset = 0;
		for (const auto &embedding : m_embeddings) {
			auto &vectors = embedding->vectors();
			const size_t size = vectors.size();
			vectors.compute_magnitudes();
			xt::view(m_mag_s, xt::range(offset, offset + size)) = vectors.magnitudes();
			offset += size;
		}
		PPK_ASSERT(offset == p_vocabulary->size());

		m_mag_t.resize({p_needle.size()});
		for (size_t j = 0; j < p_needle.size(); j++) {
			const token_t t = p_needle.token_ids()[j];
			size_t t_rel;
			const auto &t_vectors = pick_vectors(m_embeddings, t, t_rel);
			const auto row = t_vectors.unmodified(t_rel);
			m_mag_t(j) = xt::linalg::norm(row);
		}
	}

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

public:
	StaticEmbeddingMetric(
		const py::dict &p_sent_metric_def) :

		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()),
		m_needs_magnitudes(false) {
	}

	StaticEmbeddingMetric(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const py::dict &p_sent_metric_def) :

		m_embeddings(p_embeddings),
		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()),
		m_needs_magnitudes(false) {
	}

	void initialize(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);

	inline const py::dict &options() const {
		return m_options;
	}

	inline const xt::xtensor<float, 2> &similarity() const {
		return m_similarity;
	}

	inline float magnitude_s(size_t i) const {
		return m_mag_s(i);
	}

	inline float magnitude_t(size_t i) const {
		return m_mag_t(i);
	}

	inline void assert_has_magnitudes() const {
		PPK_ASSERT(m_mag_s.shape(0) > 0);
		PPK_ASSERT(m_mag_t.shape(0) > 0);
	}

	virtual MatcherFactoryRef matcher_factory() const {
		return m_matcher_factory;
	}

	virtual const std::string &name() const;

	inline const py::dict &alignment_def() const {
		return m_alignment_def;
	}
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
