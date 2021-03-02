#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

struct WordVectors {
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> V;

	V unmodified;
	V normalized;

	inline static py::array_t<float> to_py_array(const WordVectors::V &p_matrix) {
		std::vector<ssize_t> shape(2);
		shape[0] = p_matrix.rows();
		shape[1] = p_matrix.cols();
		return py::array_t<float>(
	        shape,                                      // shape
	        {shape[1] * sizeof(float), sizeof(float)},  // strides (row-major)
	        p_matrix.data());
	}

	void free_unmodified() {
		unmodified.resize(0, 0);
	}

	void free_normalized() {
		normalized.resize(0, 0);
	}

	void update_normalized() {
		normalized.resize(unmodified.rows(), unmodified.cols());
		for (Eigen::Index j = 0; j < unmodified.rows(); j++) {
			const float len = unmodified.row(j).norm();
			normalized.row(j) = unmodified.row(j) / len;
		}
	}

	py::dict to_py() const {
		py::dict d;
		d[py::str("unmodified")] = to_py_array(unmodified);
		d[py::str("normalized")] = to_py_array(normalized);
		return d;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // __VECTORIAN_WORD_VECTORS_H__
