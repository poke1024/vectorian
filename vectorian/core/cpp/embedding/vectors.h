#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

struct WordVectors {
	typedef xt::xtensor<float, 2, xt::layout_type::row_major> V;

	V unmodified;
	V normalized;

	void free_unmodified() {
		unmodified.resize({0, 0});
	}

	void free_normalized() {
		normalized.resize({0, 0});
	}

	void update_normalized() {
		normalized.resize({unmodified.shape(0), unmodified.shape(1)});
		for (size_t j = 0; j < unmodified.shape(0); j++) {
			const auto row = xt::view(unmodified, j, xt::all());
			const float len = xt::linalg::norm(row);
			xt::view(normalized, j, xt::all()) = row / len;
		}
	}

	py::dict to_py() const {
		py::dict d;
		d[py::str("unmodified")] = xt::pyarray<float>(unmodified);
		d[py::str("normalized")] = xt::pyarray<float>(normalized);
		return d;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // __VECTORIAN_WORD_VECTORS_H__
