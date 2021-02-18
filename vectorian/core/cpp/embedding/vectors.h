#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

struct WordVectors {
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> V;

	V raw;
	V normalized;

	void update_normalized() {
		normalized.resize(raw.rows(), raw.cols());
		for (Eigen::Index j = 0; j < raw.rows(); j++) {
			const float len = raw.row(j).norm();
			normalized.row(j) = raw.row(j) / len;
		}
	}
};

inline py::array_t<float> to_py_array(const WordVectors::V &p_matrix) {
	std::vector<ssize_t> shape(2);
	shape[0] = p_matrix.rows();
	shape[1] = p_matrix.cols();
	return py::array_t<float>(
        shape,                                      // shape
        {shape[1] * sizeof(float), sizeof(float)},  // strides (row-major)
        p_matrix.data());
}

#endif // __VECTORIAN_WORD_VECTORS_H__
