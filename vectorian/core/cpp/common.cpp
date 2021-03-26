#include "common.h"

py::dict to_py_array(
	const TokenVectorRef &p_array,
	const size_t p_size) {

	const std::vector<ssize_t> shape = {
		static_cast<ssize_t>(p_size)};
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
