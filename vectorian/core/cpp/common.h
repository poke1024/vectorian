#ifndef __VECTORIAN_COMMON_H__
#define __VECTORIAN_COMMON_H__

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/python/pyarrow.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <vector>
#include <set>

#define PPK_ASSERT_ENABLED 1
#include <ppk_assert.h>


namespace py = pybind11;

using Eigen::MatrixXf;
using Eigen::ArrayXf;

typedef Eigen::Map<Eigen::MatrixXf> MappedMatrixXf;
typedef Eigen::Map<Eigen::VectorXf> MappedVectorXf;

static_assert(
	std::make_tuple<int, int, int>(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION) >=
		std::make_tuple<int, int, int>(3, 3, 90),
	"Vectorian requires Eigen >= 3.3.90"); // this is a dev version as of 2021-02-15

typedef int32_t token_t;
typedef int32_t wvec_t;

#pragma pack(push, 1)
struct Token {
	token_t id;
	int32_t idx;
	int8_t len;
	int8_t pos; // universal POS tags
	int8_t tag; // Penn TreeBank style POS tags
};
#pragma pack(pop)

struct TokenSpan {
	const Token *tokens;
	int32_t len;
};

struct Slice {
	int32_t idx;
	int32_t len;

	py::tuple to_py() const {
		return py::make_tuple(idx, idx + len); // usable as slice()
	}
};

typedef Eigen::Array<token_t, Eigen::Dynamic, 1> TokenIdArray;
typedef Eigen::Map<Eigen::Array<token_t, Eigen::Dynamic, 1>> MappedTokenIdArray;

typedef std::shared_ptr<std::vector<Token>> TokenVectorRef;

typedef std::unordered_map<int, float> POSWMap;

struct MetricModifiers {
	float pos_mismatch_penalty;
	float similarity_falloff;
	float similarity_threshold;
	std::vector<float> t_pos_weights; // by index in t
};

class Query;
typedef std::shared_ptr<Query> QueryRef;
class Document;
typedef std::shared_ptr<Document> DocumentRef;
class Embedding;
typedef std::shared_ptr<Embedding> EmbeddingRef;
class Metric;
typedef std::shared_ptr<Metric> MetricRef;
class Matcher;
typedef std::shared_ptr<Matcher> MatcherRef;
class Match;
typedef std::shared_ptr<Match> MatchRef;
class ResultSet;
typedef std::shared_ptr<ResultSet> ResultSetRef;

py::array_t<float, py::array::f_style> to_py_array(MatrixXf &p_matrix);
py::array_t<token_t> to_py_array(const TokenIdArray &p_array);
py::dict to_py_array(const TokenVectorRef &p_array);
py::array_t<float, py::array::f_style> to_py_array(const MappedMatrixXf &p_matrix);
py::array_t<float> to_py_array(const MappedVectorXf &p_vector);

#define PY_ARRAY_MEMBER(STRUCT, MEMBER)                      \
	py::array_t<decltype(STRUCT::MEMBER)>(                   \
        shape,              /* shape */                      \
        {sizeof(STRUCT)},   /* strides */                    \
        reinterpret_cast<const decltype(STRUCT::MEMBER)*>(   \
            data + offsetof(STRUCT, MEMBER)))

#define ALIGNER_SLIM 1

#endif // __VECTORIAN_COMMON_H__
