#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include <xtensor/xadapt.hpp>

/*struct StaticEmbeddingTensors {
	py:array_t<float> similarity;
	py:array_t<float> mag_s;
	py:array_t<float> mag_t;
};

template<typename R, typename M>
class StaticEmbeddingTensorsReader {
	S m_similarity;
	M m_mag_s;
	M m_mag_t;

public:
	inline float similarity(size_t i, size_t j) const {
		return m_similarity(i, j);
	}

	inline float magnitude_s(size_t i) const {
		return m_mag_s(i);
	}

	inline float magnitude_t(size_t i) const {
		return m_mag_t(i);
	}
};*/

class StaticEmbeddingMetric : public Metric {
protected:
	const py::dict m_options;

	xt::pytensor<float, 2> m_similarity;
	xt::xtensor<float, 1> m_mag_s;
	xt::xtensor<float, 1> m_mag_t;
	bool m_needs_magnitudes;

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

	inline py::dict alignment_def() const {
		return m_options["alignment"].cast<py::dict>();
	}

public:
	inline StaticEmbeddingMetric(
		const py::dict &p_options) :

		m_options(p_options),
		m_needs_magnitudes(false) {
	}

	virtual void initialize(
		const QueryRef &p_query,
		const WordMetricDef &p_metric) = 0;

	inline const xt::pytensor<float, 2> &similarity() const {
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
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;

class StaticEmbeddingMetricAtom : public StaticEmbeddingMetric {

	const std::vector<StaticEmbeddingRef> m_embeddings;
	MatcherFactoryRef m_matcher_factory;

	void build_similarity_matrix(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);

	void compute_magnitudes(
		const QueryVocabularyRef &p_vocabulary,
		const Needle &p_needle) {

		m_mag_s.resize({p_vocabulary->size()});
		size_t offset = 0;
		for (const auto &embedding : m_embeddings) {
			const auto &vectors = embedding->vectors();
			const size_t size = embedding->size();

			const auto magnitudes = vectors.attr("magnitudes").cast<py::array_t<float>>();
			const auto r_mag = magnitudes.unchecked<1>();
			PPK_ASSERT(static_cast<size_t>(r_mag.shape(0)) == size);
			auto data = xt::adapt(
				const_cast<float*>(r_mag.data(0)), {r_mag.shape(0)});

			xt::view(m_mag_s, xt::range(offset, offset + size)) = data;

			offset += size;
		}
		PPK_ASSERT(offset == p_vocabulary->size());

		m_mag_t.resize({p_needle.size()});
		for (size_t j = 0; j < p_needle.size(); j++) {
			const token_t t = p_needle.token_ids()[j];
			m_mag_t(j) = m_mag_s(t);
		}
	}

public:
	StaticEmbeddingMetricAtom(
		const py::dict &p_sent_metric_def) :

		StaticEmbeddingMetric(p_sent_metric_def) {
	}

	StaticEmbeddingMetricAtom(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const py::dict &p_sent_metric_def) :

		StaticEmbeddingMetric(p_sent_metric_def),
		m_embeddings(p_embeddings) {
	}

	virtual void initialize(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);

	inline const py::dict &options() const {
		return m_options;
	}

	virtual MatcherFactoryRef matcher_factory() const {
		return m_matcher_factory;
	}

	virtual const std::string &name() const;
};

/*
class StaticEmbeddingOperator : public StaticEmbeddingMetric {
};*/

#endif // __VECTORIAN_FAST_METRIC_H__
