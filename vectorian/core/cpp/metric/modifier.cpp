#include "common.h"
#include "query.h"
#include "metric/metric.h"
#include "metric/modifier.h"

MetricRef ModifiedMetricFactory::create(
	const QueryRef &p_query) {

	py::list args;
	bool has_magnitudes = false;

	PPK_ASSERT(m_operands.size() > 0);
	const size_t num_rows = m_operands[0]->matrix()->similarity().shape(0);
	const size_t num_cols = m_operands[0]->matrix()->similarity().shape(1);

	for (const auto &operand : m_operands) {
		py::dict data;
		data["similarity"] = operand->matrix()->similarity();
		PPK_ASSERT(operand->matrix()->similarity().shape(0) == num_rows);
		PPK_ASSERT(operand->matrix()->similarity().shape(1) == num_cols);
		if (operand->matrix()->magnitudes().shape(0) > 0) {
			data["magnitudes"] = operand->matrix()->magnitudes();
			PPK_ASSERT(operand->matrix()->magnitudes().shape(0) == num_rows);
			has_magnitudes = true;
		}
		args.append(data);
	}

	const auto matrix = std::make_shared<SimilarityMatrix>();

	matrix->m_similarity.resize({ssize_t(num_rows), ssize_t(num_cols)});
	if (has_magnitudes) {
		matrix->m_magnitudes.resize({ssize_t(num_rows)});
	}

	py::dict out;
	out["similarity"] = matrix->m_similarity;
	if (has_magnitudes) {
		out["magnitudes"] = matrix->m_magnitudes;
	}
	m_operator(args, out);

	PPK_ASSERT(matrix->m_similarity.shape(0) == num_rows);
	PPK_ASSERT(matrix->m_similarity.shape(1) == num_cols);
	if (has_magnitudes) {
		PPK_ASSERT(matrix->m_magnitudes.shape(0) == num_rows);
	}

	return m_operands[0]->clone(matrix);
}
