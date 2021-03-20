#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include <xtensor/xadapt.hpp>

class StaticEmbeddingMetric : public Metric {
protected:
	const py::dict m_sent_metric_def;

	xt::pytensor<float, 2> m_similarity;
	xt::pytensor<float, 1> m_magnitudes;
	bool m_needs_magnitudes;

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

	inline py::dict alignment_def() const {
		return m_sent_metric_def["alignment"].cast<py::dict>();
	}

public:
	inline StaticEmbeddingMetric(
		const py::dict &p_sent_metric_def) :

		m_sent_metric_def(p_sent_metric_def),
		m_needs_magnitudes(false) {

		m_magnitudes.resize({0});
		PPK_ASSERT(m_magnitudes.shape(0) == 0);
	}

	inline const xt::pytensor<float, 2> &similarity() const {
		return m_similarity;
	}

	inline const xt::pytensor<float, 1> &magnitudes() const {
		return m_magnitudes;
	}

	inline void assert_has_magnitudes() const {
		PPK_ASSERT(m_magnitudes.shape(0) > 0);
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

		m_magnitudes.resize({static_cast<ssize_t>(p_vocabulary->size())});
		size_t offset = 0;
		for (const auto &embedding : m_embeddings) {
			const auto &vectors = embedding->vectors();
			const size_t size = embedding->size();

			const auto magnitudes = vectors.attr("magnitudes").cast<xt::pytensor<float, 1>>();
			xt::strided_view(m_magnitudes, {xt::range(offset, offset + size)}) = magnitudes;

			offset += size;
		}
		PPK_ASSERT(offset == p_vocabulary->size());
	}

	inline const py::dict &sent_metric_def() const {
		return m_sent_metric_def;
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

	void initialize(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);

	virtual MatcherFactoryRef matcher_factory() const {
		return m_matcher_factory;
	}

	virtual const std::string &name() const;
};

class StaticEmbeddingMetricInterpolator : public StaticEmbeddingMetric {

	const py::object m_operator;
	const std::vector<StaticEmbeddingMetricRef> m_operands;
	MatcherFactoryRef m_matcher_factory;
	std::string m_name;

public:

	StaticEmbeddingMetricInterpolator(
		const py::dict &p_sent_metric_def,
		const py::object &p_operator,
		const std::vector<StaticEmbeddingMetricRef> &p_operands) :

		StaticEmbeddingMetric(p_sent_metric_def),
		m_operator(p_operator),
		m_operands(p_operands) {
	}

	void initialize(
		const QueryRef &p_query);

	virtual MatcherFactoryRef matcher_factory() const {
		return m_matcher_factory;
	}

	virtual const std::string &name() const;
};

#endif // __VECTORIAN_FAST_METRIC_H__
