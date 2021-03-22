#include "common.h"
#include "query.h"
#include "metric/metric.h"
#include "metric/modifier.h"

SimilarityMatrixRef ModifiedSimilarityMatrixFactory::create(
	const DocumentRef &p_document) {

	py::list args;
	bool has_magnitudes = false;

	std::vector<SimilarityMatrixRef> operands;
	for (const auto &factory : m_operands) {
		operands.push_back(factory->create(p_document));
	}

	PPK_ASSERT(m_operands.size() > 0);
	const size_t num_rows = operands[0]->similarity().shape(0);
	const size_t num_cols = operands[0]->similarity().shape(1);

	for (const auto &operand : operands) {
		py::dict data;
		data["similarity"] = operand->similarity();
		PPK_ASSERT(operand->similarity().shape(0) == num_rows);
		PPK_ASSERT(operand->similarity().shape(1) == num_cols);
		if (operand->magnitudes().shape(0) > 0) {
			data["magnitudes"] = operand->magnitudes();
			PPK_ASSERT(operand->magnitudes().shape(0) == num_rows);
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

	return matrix;
}
