#ifndef __VECTORIAN_COMMON_H__
#define __VECTORIAN_COMMON_H__

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#if VECTORIAN_BLAS
#include <xtensor-blas/xlinalg.hpp>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <set>

#define PPK_ASSERT_ENABLED 1
#include <ppk_assert.h>

#define VECTORIAN_MEMORY_POOLS 0


namespace py = pybind11;

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
	int32_t offset;
	int32_t len;
};

struct Slice {
	int32_t idx;
	int32_t len;

	py::tuple to_py() const {
		return py::make_tuple(idx, idx + len); // usable as slice()
	}
};

typedef xt::xtensor<token_t, 1> TokenIdArray;
typedef std::shared_ptr<std::vector<Token>> TokenVectorRef;

typedef std::unordered_map<int, float> POSWMap;

struct MetricModifiers {
	float pos_mismatch_penalty;
	float similarity_threshold;
	std::vector<float> t_pos_weights; // by index in t
};

struct Weight {
	float flow;
	float distance;
};

struct MatcherOptions {
	bool needs_magnitudes;
	py::dict alignment_def;
};

enum EmbeddingType {
	STATIC,
	CONTEXTUAL
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
class MatcherFactory;
typedef std::shared_ptr<MatcherFactory> MatcherFactoryRef;
class Match;
typedef std::shared_ptr<Match> MatchRef;
class ResultSet;
typedef std::shared_ptr<ResultSet> ResultSetRef;
class QueryVocabulary;
typedef std::shared_ptr<QueryVocabulary> QueryVocabularyRef;
class StaticEmbedding;
typedef std::shared_ptr<StaticEmbedding> StaticEmbeddingRef;
class SimilarityMatrixBuilder;
typedef std::shared_ptr<SimilarityMatrixBuilder> SimilarityMatrixBuilderRef;

class WordMetricDef {
public:
	const std::string name;
	const std::string embedding; // e.g. fasttext
	const py::object vector_metric; // e.g. cosine
};

py::dict to_py_array(
	const TokenVectorRef &p_array, const size_t p_size);


#define PY_ARRAY_MEMBER(STRUCT, MEMBER)                      \
	py::array_t<decltype(STRUCT::MEMBER)>(                   \
        shape,              /* shape */                      \
        {sizeof(STRUCT)},   /* strides */                    \
        reinterpret_cast<const decltype(STRUCT::MEMBER)*>(   \
            data + offsetof(STRUCT, MEMBER)))

#define ALIGNER_SLIM 1


class TokenContainer {
public:
	virtual ~TokenContainer() {
	}

	virtual std::tuple<const Token*, size_t> tokens() const = 0;
};

typedef std::shared_ptr<TokenContainer> TokenContainerRef;


struct SliceStrategy {
	inline SliceStrategy() {
	}

	SliceStrategy(const py::dict &p_slice_strategy);

	std::string level; // e.g. "sentence"
	size_t window_size;
	size_t window_step;
};

typedef std::shared_ptr<SliceStrategy> SliceStrategyRef;

#endif // __VECTORIAN_COMMON_H__
