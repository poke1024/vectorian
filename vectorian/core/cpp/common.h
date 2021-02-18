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

struct Location {
	int8_t book;
	int8_t chapter;
	int8_t speaker;
	int16_t paragraph;
};

struct Sentence : public Location {
	int16_t n_tokens;
	int32_t token_at;
};

typedef int32_t token_t;

typedef Eigen::Array<token_t, Eigen::Dynamic, 1> TokenIdArray;

struct Token {
	token_t id;
	int32_t idx;
	int8_t len;
	int8_t pos; // universal POS tags
	int8_t tag; // Penn TreeBank style POS tags
};

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

#endif // __VECTORIAN_COMMON_H__
