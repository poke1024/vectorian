#ifndef __VECTORIAN_QUERY_H__
#define __VECTORIAN_QUERY_H__

#include "common.h"
#include "vocabulary.h"
#include "embedding/vectors.h"

struct TokenFilter {
	// token filters are intended as a fast way to add additional
	// filters dynamically between queries. they are not as fast
	// or efficient as using filters when creating the session.

	uint64_t pos;
	uint64_t tag;
	std::optional<xt::xtensor<bool, 1>> vocab;

	inline TokenFilter(const uint64_t p_pos, const uint64_t p_tag) :
		pos(p_pos), tag(p_tag) {
	}

	inline bool pass(const Token &t) const {
		if (vocab.has_value() && !(*vocab)(t.id)) {
			return false;
		}

		return !(((pos >> t.pos) & 1) || ((tag >> t.tag) & 1));
	}
};

typedef std::shared_ptr<TokenFilter> TokenFilterRef;


template<typename Lookup>
uint64_t parse_filter_mask(
	const py::kwargs &p_kwargs,
	const char *p_filter_name,
	Lookup lookup) {

	uint64_t filter = 0;
	if (p_kwargs && p_kwargs.contains(p_filter_name)) {
		for (auto x : p_kwargs[p_filter_name].cast<py::list>()) {
			const std::string s = x.cast<py::str>();
			const int i = lookup(s);
			if (i < 0) {
				std::ostringstream err;
				err << "illegal value " << s << " for " << p_filter_name;
				throw std::runtime_error(err.str());
			}
			filter |= static_cast<uint64_t>(1) << i;
		}
	}
	return filter;
}


class Handle {
	const py::object m_object;

public:
	inline Handle(const py::object &p_object) : m_object(p_object) {
	}

	inline const py::object &get() const {
		return m_object;
	}

	inline ~Handle() {
		m_object.attr("close")();
	}
};

typedef std::unique_ptr<Handle> HandleRef;

class VectorsCache {
	const py::object m_open;

public:
	inline VectorsCache(const py::object &p_vectors_cache) :
		m_open(p_vectors_cache.attr("open").cast<py::object>()) {
	}

	inline HandleRef open(const py::object &p_vectors_ref, const size_t p_expected_size) const {
		return std::make_unique<Handle>(m_open(p_vectors_ref, p_expected_size));
	}
};

class Query :
	public std::enable_shared_from_this<Query>,
	public ContextualVectorsContainer,
	public TokenContainer {

	const py::object m_index;
	const QueryVocabularyRef m_vocab;
	const VectorsCache m_vectors_cache;
	std::vector<MetricRef> m_metrics;
	TokenVectorRef m_t_tokens;
	py::dict m_py_t_tokens;
	float m_submatch_weight;
	bool m_bidirectional;
	TokenFilterRef m_token_filter;
	bool m_aborted;
	size_t m_max_matches;
	float m_min_score;
	POSWMap m_pos_weights;
	std::vector<float> m_t_tokens_pos_weights;
	SliceStrategy m_slice_strategy;
	std::optional<py::object> m_debug_hook;

	struct Strategy {
		EmbeddingType type;
		std::string name;
		SimilarityMatrixFactoryRef matrix_factory;
	};

	Strategy create_strategy(
		const MatcherFactoryRef &p_matcher_factory,
		const py::object &p_token_metric);

public:
	Query(
		const py::object &p_index,
		VocabularyRef p_vocab,
		const py::dict &p_contextual_embeddings) :

		m_index(p_index),
		m_vocab(std::make_shared<QueryVocabulary>(p_vocab)),
		m_vectors_cache(p_index.attr("session").attr("vectors_cache").cast<py::object>()),
		m_aborted(false) {

		for (auto item : p_contextual_embeddings) {
			m_contextual_vectors[item.first.cast<py::str>()] = item.second.cast<py::object>();
		}
	}

	void initialize(
		const py::dict &p_tokens,
		py::kwargs p_kwargs);

	virtual ~Query() {
		//std::cout << "destroying query at " << this << "\n";
	}

	const QueryVocabularyRef &vocabulary() const {
		return m_vocab;
	}

	inline py::dict py_tokens() const {
		return to_py_array(m_t_tokens, n_tokens());
	}

	inline const TokenVectorRef &tokens_vector() const {
		return m_t_tokens;
	}

	inline size_t n_tokens() const {
		return m_t_tokens->size();
	}

	virtual std::tuple<const Token*, size_t> tokens() const {
		return std::make_tuple(tokens_vector()->data(), n_tokens());
	}

	inline const POSWMap &pos_weights() const {
		return m_pos_weights;
	}

	const std::vector<MetricRef> &metrics() const {
		return m_metrics;
	}

	inline bool bidirectional() const {
		return m_bidirectional;
	}

	inline const TokenFilterRef &token_filter() const {
	    return m_token_filter;
	}

	ResultSetRef match(
		const DocumentRef &p_document);

	bool aborted() const {
		return m_aborted;
	}

	void abort() {
		m_aborted = true;
	}

	inline size_t max_matches() const {
		return m_max_matches;
	}

	inline float min_score() const {
		return m_min_score;
	}

	inline float submatch_weight() const {
		return m_submatch_weight;
	}

	inline const SliceStrategy& slice_strategy() const {
		return m_slice_strategy;
	}

	inline const std::optional<py::object>& debug_hook() const {
		return m_debug_hook;
	}

	template<typename Slice>
	py::dict make_py_debug_slice(const Slice &p_slice) const {
		const QueryVocabularyRef vocab = this->vocabulary();

		const auto token_vector = [&] (const auto &get_id, const int n) {
			py::list id;
			py::list text;
			for (int i = 0; i < n; i++) {
				id.append(get_id(i));
				text.append(vocab->id_to_token(get_id(i)));
			}
			py::dict tokens;
			tokens["id"] = id;
			tokens["text"] = text;
			return tokens;
		};

		py::dict data;

		data["s"] = token_vector([&] (int i) {
			return p_slice.s(i).id;
		}, p_slice.len_s());

		data["t"] = token_vector([&] (int i) {
			return p_slice.t(i).id;
		}, p_slice.len_t());

		return data;
	}

	inline const py::object &index() const {
		return m_index;
	}

	inline const VectorsCache &vectors_cache() const {
		return m_vectors_cache;
	}

	TokenFilterRef make_token_filter(
		const py::kwargs &p_kwargs) const;
};

typedef std::shared_ptr<Query> QueryRef;

#endif // __VECTORIAN_QUERY_H__
