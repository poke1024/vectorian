#include "common.h"

py::array_t<float, py::array::f_style> to_py_array(MatrixXf &p_matrix) {
	std::vector<ssize_t> shape(2);
	shape[0] = p_matrix.rows();
	shape[1] = p_matrix.cols();
	return py::array_t<float, py::array::f_style>(
        shape,                                      // shape
        {shape[0] * sizeof(float), sizeof(float)},  // strides (col-major)
        p_matrix.data());
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
