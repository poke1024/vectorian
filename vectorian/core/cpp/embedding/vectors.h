#ifndef __VECTORIAN_WORD_VECTORS_H__
#define __VECTORIAN_WORD_VECTORS_H__

#include "common.h"

struct WordVectors {
	typedef xt::xtensor<float, 2, xt::layout_type::row_major> V;

	V unmodified;
	V normalized;
	xt::xtensor<float, 1> magnitudes;

	void free_unmodified() {
		unmodified.resize({0, 0});
	}

	void free_normalized() {
		normalized.resize({0, 0});
	}

	void update_normalized() {
		compute_magnitudes();

		constexpr float eps = std::numeric_limits<float>::epsilon() * 100.0f;
		normalized.resize({unmodified.shape(0), unmodified.shape(1)});
		for (size_t i = 0; i < unmodified.shape(0); i++) {
			const float len = magnitudes(i);
			if (len > eps) {
				const auto row = xt::view(unmodified, i, xt::all());
				xt::view(normalized, i, xt::all()) = row / len;
			} else {
				xt::view(normalized, i, xt::all()).fill(0.0f);
			}
		}
	}

	void compute_magnitudes() {
		const size_t n = unmodified.shape(0);
		if (magnitudes.shape(0) == n) {
			return;
		}
		magnitudes.resize({n});

		for (size_t i = 0; i < n; i++) {
			const auto row = xt::view(unmodified, i, xt::all());
			magnitudes(i) = xt::linalg::norm(row);
		}
	}

	py::dict to_py() const {
		py::dict d;
		d[py::str("unmodified")] = xt::pyarray<float>(unmodified);
		d[py::str("normalized")] = xt::pyarray<float>(normalized);
		return d;
	}
};

#endif // __VECTORIAN_WORD_VECTORS_H__
