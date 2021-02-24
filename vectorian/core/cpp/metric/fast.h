#ifndef __VECTORIAN_FAST_METRIC_H__
#define __VECTORIAN_FAST_METRIC_H__

#include "metric/metric.h"

class FastMetric : public Metric {
protected:
	const EmbeddingRef m_embedding;
	const py::dict m_options;
	const py::dict m_alignment_def;
	MatrixXf m_similarity;

public:
	FastMetric(
		const EmbeddingRef &p_embedding,
		const py::dict &p_sent_metric_def) :

		m_embedding(p_embedding),
		m_options(p_sent_metric_def),
		m_alignment_def(m_options["alignment"].cast<py::dict>()) {
	}

	inline const py::dict &options() const {
		return m_options;
	}

	inline MatrixXf &w_similarity() {
		return m_similarity;
	}

	inline const MatrixXf &similarity() const {
		return m_similarity;
	}

	virtual MatcherRef create_matcher(
		const QueryRef &p_query,
		const DocumentRef &p_document);

	virtual const std::string &name() const;

	inline const py::dict &alignment_def() const {
		return m_alignment_def;
	}
};

typedef std::shared_ptr<FastMetric> FastMetricRef;

#endif // __VECTORIAN_FAST_METRIC_H__
