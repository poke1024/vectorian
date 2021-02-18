#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

struct WordVectors {
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> V;
	typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> R;

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

#endif // __VECTORIAN_WORD_VECTORS_H__
