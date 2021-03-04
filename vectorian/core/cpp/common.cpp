#include "common.h"

py::array_t<float, py::array::f_style> to_py_array(const MatrixXf &p_matrix) {
	std::vector<ssize_t> shape(2);
	shape[0] = p_matrix.rows();
	shape[1] = p_matrix.cols();
	return py::array_t<float, py::array::f_style>(
        shape,                                      // shape
        {shape[0] * sizeof(float), sizeof(float)},  // strides (col-major)
        p_matrix.data());
}

py::array_t<float> to_py_array(const VectorXf &p_vector) {
	std::vector<ssize_t> shape(1);
	shape[0] = p_vector.rows();
	return py::array_t<float>(
        shape,                                      // shape
        {sizeof(float)},                            // strides
        p_vector.data());
}

py::array_t<float> to_py_array(const MappedMatrixXf &p_matrix) {
#if 1
	const auto m = p_matrix.rows();
	const auto n = p_matrix.cols();
	py::array_t<float> array({ m, n });
	auto r = array.mutable_unchecked<2>();
	for (Eigen::Index i = 0; i < m; i++) {
		for (Eigen::Index j = 0; j < n; j++) {
			r(i, j) = p_matrix(i, j);
		}
	}
	return array;
#else
	std::vector<ssize_t> shape(2);
	shape[0] = p_matrix.rows();
	shape[1] = p_matrix.cols();
	return py::array_t<float, py::array::f_style>(
        shape,                                      // shape
        {p_matrix.outerStride() * sizeof(float), p_matrix.innerStride() * sizeof(float)},  // strides (col-major)
        p_matrix.data());
#endif
}

py::array_t<float> to_py_array(const MappedVectorXf &p_vector) {
#if 1
	const auto n = p_vector.rows();
	py::array_t<float> array(n);
	auto r = array.mutable_unchecked<1>();
	for (Eigen::Index i = 0; i < n; i++) {
		r(i) = p_vector(i);
	}
	return array;
#else
	std::vector<ssize_t> shape(1);
	shape[0] = p_vector.rows();
	return py::array_t<float>(
        shape,              // shape
        {p_vector.innerStride() * sizeof(float)},    // strides
        p_vector.data());   // data pointer
#endif
}

py::array_t<token_t> to_py_array(const TokenIdArray &p_array) {
	std::vector<ssize_t> shape(1);
	shape[0] = p_array.rows();
	return py::array_t<token_t>(
        shape,              // shape
        {sizeof(token_t)},  // strides
        p_array.data());    // data pointer
}

py::dict to_py_array(const TokenVectorRef &p_array) {

	const std::vector<ssize_t> shape = {
		static_cast<ssize_t>(p_array->size())};
	const uint8_t* const data =
		reinterpret_cast<const uint8_t*>(p_array->data());

	py::dict d;

	d["id"] = PY_ARRAY_MEMBER(Token, id);
	d["idx"] = PY_ARRAY_MEMBER(Token, idx);
	d["len"] = PY_ARRAY_MEMBER(Token, len);
	d["pos"] = PY_ARRAY_MEMBER(Token, pos);
	d["tag"] = PY_ARRAY_MEMBER(Token, tag);

	return d;
}
