#include <iostream>

#include "common.h"
#include "match/region.h"
#include "match/match.h"
#include "match/match_impl.h"
#include "embedding/embedding.h"
#include "embedding/static.h"
#include "vocabulary.h"
#include "query.h"
#include "document.h"
#include "result_set.h"

namespace py = pybind11;

void init_pyarrow() {
	if (arrow::py::import_pyarrow() != 0) {
		std::cerr << "error initializing pyarrow.\n";
	}
}

void run_sanity_checks() {
	{
		TokenIdArray a;
		a.resize(4);
		for (int i = 0; i < a.rows(); i++) {
			a(i) = 2.5 * i;
		}

		auto array = to_py_array(a);
		for (int i = 0; i < a.rows(); i++) {
			PPK_ASSERT(array(i).cast<float>() == a(i));
		}
	}

	{
		WordVectors::V v;
		v.resize(3, 10);
		for (int i = 0; i < v.rows(); i++) {
			for (int j = 0; j < v.cols(); j++) {
				v(i, j) = 7.5 * i + j * 2.25;
			}
		}
		auto array = to_py_array(v);
		for (int i = 0; i < v.rows(); i++) {
			for (int j = 0; j < v.cols(); j++) {
				PPK_ASSERT(array(i, j).cast<float>() == v(i, j));
			}
		}
	}
}

py::str backend_build_time() {
	return __TIMESTAMP__;
}

// !!!
// caution: name in PYBIND11_MODULE below needs to match filename
// !!!
PYBIND11_MODULE(core, m) {
	m.def("init_pyarrow", &init_pyarrow);
	m.def("run_sanity_checks", &run_sanity_checks);
	m.def("backend_build_time", &backend_build_time);

	py::class_<Region, RegionRef> region(m, "Region");
	region.def_property_readonly("s", &Region::s);
	region.def_property_readonly("mismatch_penalty", &Region::mismatch_penalty);
	region.def_property_readonly("matched", &Region::is_matched);

	py::class_<MatchedRegion, Region, MatchedRegionRef> matched_region(m, "MatchedRegion");
	matched_region.def_property_readonly("t", &MatchedRegion::t);
	matched_region.def_property_readonly("similarity", &MatchedRegion::similarity);
	matched_region.def_property_readonly("weight", &MatchedRegion::weight);
	matched_region.def_property_readonly("pos_s", &MatchedRegion::pos_s);
	matched_region.def_property_readonly("pos_t", &MatchedRegion::pos_t);
	matched_region.def_property_readonly("metric", &MatchedRegion::metric);

	py::class_<ExternalMatcher, ExternalMatcherRef> matcher(m, "ExternalMatcher");
	matcher.def(py::init<const QueryRef&,
		const DocumentRef&,
		const MetricRef&>());

	py::class_<Match, MatchRef> match(m, "Match");
	match.def(py::init<const MatcherRef&,
		const DocumentRef&,
		const int32_t,
		const std::vector<int16_t>&,
		const float>());
	match.def_property_readonly("query", &Match::query);
	match.def_property_readonly("document", &Match::document);
	match.def_property_readonly("assignment", &Match::py_assignment);
	match.def_property_readonly("score", &Match::score);
	match.def_property_readonly("metric", &Match::metric_name);
	match.def_property_readonly("sentence", &Match::sentence_id);
	match.def_property_readonly("regions", &Match::regions);
	match.def_property_readonly("omitted", &Match::omitted);

	py::class_<Embedding, EmbeddingRef> embedding(m, "Embedding");

	py::class_<StaticEmbedding, Embedding, StaticEmbeddingRef> fast_embedding(m, "StaticEmbedding");
	fast_embedding.def(py::init<const std::string &, py::object>());
	//fast_embedding.def("cosine_similarity", &StaticEmbedding::cosine_similarity);
	//fast_embedding.def("similarity_matrix", &StaticEmbedding::similarity_matrix);
	//fast_embedding.def("load_percentiles", &StaticEmbedding::load_percentiles);
	fast_embedding.def_property_readonly("n_tokens", &StaticEmbedding::n_tokens);
	fast_embedding.def_property_readonly("measures", &StaticEmbedding::measures);

	py::class_<Vocabulary, VocabularyRef> vocabulary(m, "Vocabulary");
	vocabulary.def(py::init());
	vocabulary.def_property_readonly("size", &Vocabulary::size);
	vocabulary.def("token_to_id", &Vocabulary::token_to_id);
	vocabulary.def("id_to_token", &Vocabulary::id_to_token);
	vocabulary.def("add_embedding", &Vocabulary::add_embedding);

	py::class_<Query, QueryRef> query(m, "Query");
	query.def(py::init<VocabularyRef, const std::string &, py::handle, py::kwargs>());
	query.def_property_readonly("text", &Query::text);
	query.def("substr", &Query::substr);
	query.def_property_readonly("tokens", &Query::py_tokens);
	query.def("abort", &Query::abort);

	py::class_<Document, DocumentRef> document(m, "Document");
	document.def(py::init<
		py::object, int64_t, VocabularyRef, const std::string&, py::object,
		py::object, py::dict, const std::string&>());
	document.def("find", &Document::find);
	document.def("__str__", &Document::__str__);
	document.def("__repr__", &Document::__str__);
	document.def_property_readonly("id", &Document::id);
	document.def_property_readonly("text", &Document::text);
	document.def("substr", &Document::substr);
	document.def_property_readonly("tokens", &Document::py_tokens);
	document.def_property_readonly("path", &Document::path);
	document.def_property_readonly("metadata", &Document::metadata);
	document.def("sentence_info", &Document::py_sentence_info);
	document.def("sentence", &Document::py_sentence);
	document.def_property_readonly("sentences", &Document::py_sentences);
	document.def_property_readonly("n_tokens", &Document::n_tokens);
	document.def_property_readonly("n_sentences", &Document::n_sentences);
	document.def_property_readonly("max_sentence_len", &Document::max_len_s);

	py::class_<ResultSet, ResultSetRef> result_set(m, "ResultSet");
	result_set.def_property_readonly("size", &ResultSet::size);
	result_set.def("best_n", &ResultSet::best_n);
	result_set.def("extend", &ResultSet::extend);

	py::class_<ExternalMetric, ExternalMetricRef> ext_metric(m, "ExternalMetric");
	ext_metric.def(py::init<const std::string&>());

	/*py::class_<LargeMatrix, LargeMatrixRef> matrix(m, "LargeMatrix");
	matrix.def(py::init<const std::string &>());
	matrix.def("create", &LargeMatrix::create);
	matrix.def("close", &LargeMatrix::close);
	matrix.def("write", &LargeMatrix::write);
	*/
}
