#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "common.h"
#include "match/region.h"
#include "match/match.h"
#include "match/match_impl.h"
#include "embedding/embedding.h"
#include "embedding/static.h"
#include "embedding/contextual.h"
#include "vocabulary.h"
#include "query.h"
#include "document.h"
#include "result_set.h"

namespace py = pybind11;

py::str backend_build_time() {
	return __TIMESTAMP__;
}

// !!!
// caution: name in PYBIND11_MODULE below needs to match filename
// !!!
#if VECTORIAN_SETUP_PY
PYBIND11_MODULE(vectorian_core, m) {
#else
PYBIND11_MODULE(core, m) {
#endif
	xt::import_numpy();

	m.def("backend_build_time", &backend_build_time);

	py::class_<Region, RegionRef> region(m, "Region");
	region.def_property_readonly("s", &Region::s);
	region.def_property_readonly("mismatch_penalty", &Region::mismatch_penalty);
	region.def_property_readonly("matched", &Region::is_matched);

	py::class_<MatchedRegion, Region, MatchedRegionRef> matched_region(m, "MatchedRegion");
	matched_region.def_property_readonly("num_edges", &MatchedRegion::num_edges);
	matched_region.def("flow", &MatchedRegion::flow);
	matched_region.def("distance", &MatchedRegion::distance);
	matched_region.def("t", &MatchedRegion::t);
	matched_region.def_property_readonly("pos_s", &MatchedRegion::pos_s);
	matched_region.def("pos_t", &MatchedRegion::pos_t);
	matched_region.def("metric", &MatchedRegion::metric);

	py::class_<ExternalMatcher, ExternalMatcherRef> matcher(m, "ExternalMatcher");
	matcher.def(py::init<const QueryRef&,
		const DocumentRef&,
		const MetricRef&>());

	py::class_<Match, MatchRef> match(m, "Match");
	match.def(py::init<const MatcherRef&,
		const DocumentRef&,
		const int32_t,
		const FlowRef<int16_t>&,
		const float>());
	match.def_property_readonly("query", &Match::query);
	match.def_property_readonly("document", &Match::document);
	match.def_property_readonly("flow", &Match::flow_to_py);
	match.def_property_readonly("score", &Match::score);
	match.def_property_readonly("metric", &Match::metric_name);
	match.def_property_readonly("slice_id", &Match::slice_id);
	match.def("regions", &Match::regions);
	match.def_property_readonly("omitted", &Match::omitted);

	py::class_<EmbeddingManager, EmbeddingManagerRef> embedding_manager(m, "EmbeddingManager");
	embedding_manager.def(py::init<>());
	embedding_manager.def("add_embedding", &EmbeddingManager::add_embedding);
	embedding_manager.def("to_index", &EmbeddingManager::to_index);
	embedding_manager.def("compile_static", &EmbeddingManager::compile_static);
	embedding_manager.def("compile_contextual", &EmbeddingManager::compile_contextual);

	py::class_<Embedding, EmbeddingRef> embedding(m, "Embedding");

	py::class_<StaticEmbedding, Embedding, StaticEmbeddingRef> static_embedding(m, "StaticEmbedding");
	static_embedding.def(py::init<py::object, py::list>());
	static_embedding.def_property_readonly("vectors", &StaticEmbedding::py_vectors);
	static_embedding.def_property_readonly("size", &StaticEmbedding::size);

	py::class_<ContextualEmbedding, Embedding, ContextualEmbeddingRef> contextual_embedding(m, "ContextualEmbedding");
	contextual_embedding.def(py::init<const std::string&>());

	py::class_<Vocabulary, VocabularyRef> vocabulary(m, "Vocabulary");
	vocabulary.def(py::init<const EmbeddingManagerRef&>());
	vocabulary.def_property_readonly("size", &Vocabulary::size);
	vocabulary.def("token_to_id", &Vocabulary::token_to_id);
	vocabulary.def("id_to_token", &Vocabulary::id_to_token);
	vocabulary.def("compile_embeddings", &Vocabulary::compile_embeddings);

	py::class_<Frequencies, FrequenciesRef> frequencies(m, "Frequencies");
	frequencies.def(py::init<const VocabularyRef&>());
	frequencies.def("add", &Frequencies::add);
	frequencies.def("tf", &Frequencies::tf);
	frequencies.def("df", &Frequencies::df);
	frequencies.def("tf_idf", &Frequencies::tf_idf);
	frequencies.def("tf_tensor", &Frequencies::tf_tensor);
	frequencies.def("df_tensor", &Frequencies::df_tensor);
	frequencies.def("tf_idf_tensor", &Frequencies::tf_idf_tensor);

	py::class_<Query, QueryRef> query(m, "Query");
	query.def(py::init<const py::object&, VocabularyRef, const py::dict&>());
	query.def("initialize", &Query::initialize);
	query.def_property_readonly("n_tokens", &Query::n_tokens);

	query.def_property_readonly("tokens", &Query::py_tokens);
	query.def("abort", &Query::abort);

	py::class_<Document, DocumentRef> document(m, "Document");
	document.def(py::init<
		int64_t, VocabularyRef, py::dict, py::dict, py::dict, py::dict>());
	document.def("find", &Document::find);
	document.def("__str__", &Document::__str__);
	document.def("__repr__", &Document::__str__);
	document.def_property_readonly("id", &Document::id);
	document.def_property_readonly("tokens", &Document::py_tokens);
	document.def_property_readonly("path", &Document::path);
	document.def_property_readonly("metadata", &Document::metadata);
	document.def_property_readonly("n_tokens", &Document::n_tokens);
	document.def("max_len", &Document::max_len);

	py::class_<ResultSet, ResultSetRef> result_set(m, "ResultSet");
	result_set.def_property_readonly("size", &ResultSet::size);
	result_set.def("best_n", &ResultSet::best_n);
	result_set.def("extend", &ResultSet::extend);

	//py::class_<ExternalMetric, ExternalMetricRef> ext_metric(m, "ExternalMetric");
	//ext_metric.def(py::init<const std::string&>());
}
