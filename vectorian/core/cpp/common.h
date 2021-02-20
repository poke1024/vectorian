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

#include <ppk_assert.h>


namespace py = pybind11;

using Eigen::MatrixXf;
using Eigen::ArrayXf;

static_assert(
	std::make_tuple<int, int, int>(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION) >=
		std::make_tuple<int, int, int>(3, 3, 90),
	"Vectorian requires Eigen >= 3.3.90"); // this is a dev version as of 2021-02-15

typedef int32_t token_t;

#pragma pack(push, 1)
struct Location {
	int16_t paragraph;
	int8_t book;
	int8_t chapter;
	int8_t speaker;
};

struct Sentence : public Location {
	int32_t token_at;
	int16_t n_tokens;
};

struct Token {
	token_t id;
	int32_t idx;
	int8_t len;
	int8_t pos; // universal POS tags
	int8_t tag; // Penn TreeBank style POS tags
};
#pragma pack(pop)

typedef Eigen::Array<token_t, Eigen::Dynamic, 1> TokenIdArray;

typedef std::shared_ptr<std::vector<Token>> TokenVectorRef;

typedef std::unordered_map<int, float> POSWMap;

struct MetricModifiers {
	float pos_mismatch_penalty;
	float similarity_falloff;
	float similarity_threshold;
	POSWMap pos_weights;
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

#define PY_ARRAY_MEMBER(STRUCT, MEMBER)                      \
	py::array_t<decltype(STRUCT::MEMBER)>(                   \
        shape,              /* shape */                      \
        {sizeof(STRUCT)},   /* strides */                    \
        reinterpret_cast<const decltype(STRUCT::MEMBER)*>(   \
            data + offsetof(STRUCT, MEMBER)))

#define ALIGNER_SLIM 1

#endif // __VECTORIAN_COMMON_H__
