#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include <xtensor/xadapt.hpp>

class StaticEmbeddingMetric : public Metric {
public:
	inline StaticEmbeddingMetric(
		const std::string &p_name,
		const SimilarityMatrixRef &p_matrix,
		const MatcherFactoryRef &p_matcher_factory) :

		Metric(p_name, p_matrix, p_matcher_factory) {
	}

	virtual MetricRef clone(const SimilarityMatrixRef &p_matrix) {
		return std::make_shared<StaticEmbeddingMetric>(
			m_name, p_matrix, m_matcher_factory);
	}
};

typedef std::shared_ptr<StaticEmbeddingMetric> StaticEmbeddingMetricRef;

class StaticEmbeddingMetricFactory {

	py::dict m_sent_metric_def;
	const std::vector<StaticEmbeddingRef> m_embeddings;

	inline py::dict alignment_def() const {
		return m_sent_metric_def["alignment"].cast<py::dict>();
	}

	SimilarityMatrixRef build_similarity_matrix(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);

	void compute_magnitudes(
		const SimilarityMatrixRef &p_matrix,
		const QueryVocabularyRef &p_vocabulary,
		const Needle &p_needle) {

		p_matrix->m_magnitudes.resize({static_cast<ssize_t>(p_vocabulary->size())});
		size_t offset = 0;
		for (const auto &embedding : m_embeddings) {
			const auto &vectors = embedding->vectors();
			const size_t size = embedding->size();

			const auto magnitudes = vectors.attr("magnitudes").cast<xt::pytensor<float, 1>>();
			xt::strided_view(p_matrix->m_magnitudes, {xt::range(offset, offset + size)}) = magnitudes;

			offset += size;
		}
		PPK_ASSERT(offset == p_vocabulary->size());
	}

	MatcherFactoryRef create_matcher_factory(
		const QueryRef &p_query);

	inline const py::dict &sent_metric_def() const {
		return m_sent_metric_def;
	}

public:
	StaticEmbeddingMetricFactory(
		const py::dict &p_sent_metric_def) :

		m_sent_metric_def(p_sent_metric_def) {
	}

	StaticEmbeddingMetricFactory(
		const std::vector<StaticEmbeddingRef> &p_embeddings,
		const py::dict &p_sent_metric_def) :

		m_sent_metric_def(p_sent_metric_def),
		m_embeddings(p_embeddings) {
	}

	StaticEmbeddingMetricRef create(
		const QueryRef &p_query,
		const WordMetricDef &p_metric);
};

#endif // __VECTORIAN_FAST_METRIC_H__
