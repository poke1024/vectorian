#ifndef __VECTORIAN_UTILS_H__
#define __VECTORIAN_UTILS_H__

#include "common.h"

#if PYARROW_0_12_1
inline auto column_data(const std::shared_ptr<arrow::Column> &c) {
	return c->data();
}
#else
inline auto column_data(const std::shared_ptr<arrow::ChunkedArray> &c) {
	return c;
}
#endif

template<typename ArrowType, typename Array>
void ensure_type(const Array &array) {
    if (array->type_id() != arrow::TypeTraits<ArrowType>::type_singleton()->id()) {
        std::string got = array->type()->name();
        std::string expected = arrow::TypeTraits<ArrowType>::type_singleton()->name();

        std::ostringstream err;
        err << "parquet data type of chunk is wrong. expected " <<
            expected << ", got " + got + ".";

        throw std::runtime_error(err.str());
    }
}

inline std::shared_ptr<arrow::Table> unwrap_table(PyObject *p_table) {
	arrow::Result<std::shared_ptr<arrow::Table>> table(
        arrow::py::unwrap_table(p_table));
	if (!table.ok()) {
    	std::ostringstream err;
    	err << "PyObject of type " << Py_TYPE(p_table)->tp_name <<
    	    " could not get converted to a pyarrow table";
		throw std::runtime_error(err.str());
	}
	return *table;
}

inline std::shared_ptr<arrow::Table> unwrap_table(const py::object &p_table) {
    return unwrap_table(p_table.ptr());
}

template<typename T, typename TV>
const std::vector<TV> numeric_column(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));

	std::vector<TV> values;
    int64_t count = 0;
    for (int64_t k = 0; k < data->num_chunks(); k++) {
    	auto array = data->chunk(k);
        count += array->length();
    }

    values.resize(count);
    int64_t write_offset = 0;

    for (int64_t k = 0; k < data->num_chunks(); k++) {
    	auto array = data->chunk(k);

        if (array->type_id() != arrow::TypeTraits<T>::type_singleton()->id()) {
            std::stringstream s;
            s << "extract_raw_values: wrong data type " <<
                array->type()->name() << " != " << arrow::TypeTraits<T>::type_singleton()->name();
            throw std::runtime_error(s.str());
        }

    	auto num_array = std::static_pointer_cast<arrow::NumericArray<T>>(array);
    	const auto *raw = num_array->raw_values();
    	std::copy(raw, raw + array->length(), &values[write_offset]);
    	write_offset += array->length();
    }

    return values;
}

template<typename F>
class StringVisitor : public arrow::ArrayVisitor {
	F f;
	size_t m_index;

public:
	StringVisitor(const F &f) : f(f), m_index(0) {
	}

	virtual arrow::Status Visit(const arrow::StringArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.GetString(i));
		}
		return arrow::Status::OK();
	}
};

template<typename F>
void iterate_strings(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field,
	const F &f) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));
    StringVisitor v(f);
    for (int64_t k = 0; k < data->num_chunks(); k++) {
        auto array = data->chunk(k);
        ensure_type<arrow::StringType>(array);

        if (!array->Accept(&v).ok()) {
            throw std::runtime_error("arrow iteration error in iterate_strings");
        }
    }
}

template<typename F>
class FloatVisitor : public arrow::ArrayVisitor {
	F f;
	size_t m_index;

public:
	FloatVisitor(const F &f) : f(f), m_index(0) {
	}

	virtual arrow::Status Visit(const arrow::FloatArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.Value(i));
		}
		return arrow::Status::OK();
	}

	virtual arrow::Status Visit(const arrow::DoubleArray &array) {
		const int n = array.length();
		for (int i = 0; i < n; i++) {
			f(m_index++, array.Value(i));
		}
		return arrow::Status::OK();
	}
};

template<typename F>
void iterate_floats(
	const std::shared_ptr<arrow::Table> &table,
	const std::string &field,
	const F &f) {

	const int i = table->schema()->GetFieldIndex(field);
	if (i < 0) {
		throw std::runtime_error("extract_raw_values: illegal field name");
	}
	auto data = column_data(table->column(i));
	FloatVisitor v(f);
    for (int64_t k = 0; k < data->num_chunks(); k++) {
        auto array = data->chunk(k);
        ensure_type<arrow::DoubleType>(array);
        if (!array->Accept(&v).ok()) {
            throw std::runtime_error("arrow iteration error in iterate_floats");
        }
    }
}

template<typename ArrowType, typename CType, typename F>
void for_each_column(
	const std::shared_ptr<arrow::Table> &p_table, const F &p_f, int p_first_column = 0) {

	for (int i = p_first_column; i < p_table->num_columns(); i++) {
		const auto data = column_data(p_table->column(i));
		size_t offset = 0;

	    for (int64_t k = 0; k < data->num_chunks(); k++) {
            auto array = data->chunk(k);
            ensure_type<ArrowType>(array);

            auto num_array = std::static_pointer_cast<arrow::NumericArray<ArrowType>>(array);

            const Eigen::Map<Eigen::Array<CType, Eigen::Dynamic, 1>> v(
                const_cast<CType*>(num_array->raw_values()), num_array->length());

            p_f(i, v, offset);

            offset += num_array->length();
	    }
	}
}

#endif // __VECTORIAN_UTILS_H__
