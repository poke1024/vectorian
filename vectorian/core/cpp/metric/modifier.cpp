#include "common.h"
#include "query.h"
#include "metric/metric.h"
#include "metric/modifier.h"

ModifiedSimilarityMatrixFactory::ModifiedSimilarityMatrixFactory(
	const py::object &p_operator,
	const std::vector<SimilarityMatrixFactoryRef> &p_operands) :

	m_operator(p_operator),
	m_operands(p_operands),

	PY_SIMILARITY("similarity"),
	PY_MAGNITUDES_S("magnitudes_s"),
	PY_MAGNITUDES_T("magnitudes_t"){
}

SimilarityMatrixRef ModifiedSimilarityMatrixFactory::create(
	const EmbeddingType p_embedding_type,
	const DocumentRef &p_document) {

	py::list args;
	bool has_magnitudes = false;

	std::vector<SimilarityMatrixRef> operands;
	for (const auto &factory : m_operands) {
		operands.push_back(factory->create(
			p_embedding_type, p_document));
	}

	PPK_ASSERT(m_operands.size() > 0);
	const size_t num_rows = operands[0]->sim().shape(0);
	const size_t num_cols = operands[0]->sim().shape(1);

	for (const auto &operand : operands) {
		py::dict data;
		data[PY_SIMILARITY] = operand->sim();
		PPK_ASSERT(operand->sim().shape(0) == num_rows);
		PPK_ASSERT(operand->sim().shape(1) == num_cols);
		if (operand->mag_s().shape(0) > 0) {
			data[PY_MAGNITUDES_S] = operand->mag_s();
			data[PY_MAGNITUDES_T] = operand->mag_t();
			PPK_ASSERT(operand->mag_s().shape(0) == num_rows);
			PPK_ASSERT(operand->mag_t().shape(0) == num_cols);
			has_magnitudes = true;
		}
		args.append(data);
	}

	const auto matrix = operands[0]->clone_empty();

	matrix->m_similarity.resize({ssize_t(num_rows), ssize_t(num_cols)});
	if (has_magnitudes) {
		matrix->m_magnitudes_s.resize({ssize_t(num_rows)});
		matrix->m_magnitudes_t.resize({ssize_t(num_cols)});
	}

	py::dict out;
	out[PY_SIMILARITY] = matrix->m_similarity;
	if (has_magnitudes) {
		out[PY_MAGNITUDES_S] = matrix->m_magnitudes_s;
		out[PY_MAGNITUDES_T] = matrix->m_magnitudes_t;
	}
	m_operator(args, out);

	PPK_ASSERT(matrix->m_similarity.shape(0) == num_rows);
	PPK_ASSERT(matrix->m_similarity.shape(1) == num_cols);
	if (has_magnitudes) {
		PPK_ASSERT(matrix->m_magnitudes_s.shape(0) == num_rows);
		PPK_ASSERT(matrix->m_magnitudes_t.shape(0) == num_cols);
	}

	return matrix;
}
