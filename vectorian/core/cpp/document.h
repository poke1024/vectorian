#ifndef __VECTORIAN_DOCUMENT_H__
#define __VECTORIAN_DOCUMENT_H__

#include "common.h"
#include "vocabulary.h"
#include "embedding/vectors.h"

inline void add_dummy_token(std::vector<Token> &tokens) {
	Token t;
	t.id = -1;
	t.len = 0;
	t.pos = -1;
	t.tag = -1;

	if (tokens.empty()) {
		t.idx = 0;
	} else {
		t.idx = tokens.rbegin()->idx + tokens.rbegin()->len;
	}

	// adding a last dummy token with the correct idx is handy.
	tokens.push_back(t);
}

class FixedSpans {
	const size_t m_size;

public:
	FixedSpans(const size_t p_size) : m_size(p_size) {
	}

	inline size_t size() const {
		return m_size;
	}

	inline int32_t start(const size_t p_index) const {
		return p_index;
	}

	inline int32_t end(const size_t p_index) const {
		return p_index + 1;
	}

	inline int32_t len(const size_t p_index) const {
		return 1;
	}

	inline int32_t bounded_len(const size_t p_index, const size_t p_size) const {
		return std::min(p_size, m_size - p_index);
	}

	inline int32_t max_len(const size_t p_window_size) const {
		return p_window_size;
	}
};


class VariableSpans {
public:
	typedef int32_t offset_t;

	struct Span {
		offset_t start;
		offset_t end;
	};

private:
	const std::vector<Span> m_spans;
	offset_t m_max_len;

public:
	VariableSpans(std::vector<Span> &&p_spans) : m_spans(p_spans) {
		offset_t max_len = 0;
		for (const auto &s : m_spans) {
			max_len = std::max(max_len, s.end - s.start);
		}
		m_max_len = max_len;
	}

	inline size_t size() const {
		return m_spans.size();
	}

	inline offset_t start(const size_t p_index) const {
		return m_spans[p_index].start;
	}

	inline offset_t end(const size_t p_index) const {
		return m_spans[p_index].end;
	}

	inline offset_t len(const size_t p_index) const {
		return end(p_index) - start(p_index);
	}

	inline offset_t bounded_len(const size_t p_index, const size_t p_size) const {
		const size_t i1 = std::min(p_index + p_size - 1, m_spans.size() - 1);
		return end(i1) - start(p_index);
	}

	inline offset_t max_len(const size_t p_window_size) const {
		return m_max_len * p_window_size;
	}
};

class Spans {
	std::optional<FixedSpans> m_fixed;
	std::optional<VariableSpans> m_variable;

public:
	Spans(FixedSpans &&p_spans) : m_fixed(p_spans) {
	}

	Spans(VariableSpans &&p_spans) : m_variable(p_spans) {
	}

	inline size_t size() const {
		return m_variable.has_value() ? (*m_variable).size() : (*m_fixed).size();
	}

	inline int32_t start(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).start(p_index) : (*m_fixed).start(p_index);
	}

	inline int32_t end(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).end(p_index) : (*m_fixed).end(p_index);
	}

	inline int32_t len(const size_t p_index) const {
		return m_variable.has_value() ? (*m_variable).len(p_index) : (*m_fixed).len(p_index);
	}

	inline int32_t bounded_len(const size_t p_index, const size_t p_size) const {
		return m_variable.has_value() ?
			(*m_variable).bounded_len(p_index, p_size) :
			(*m_fixed).bounded_len(p_index, p_size);
	}

	inline Slice slice(const size_t p_index) const {
		return Slice{start(p_index), len(p_index)};
	}

	inline int32_t max_len(const size_t p_window_size) const {
		return m_variable.has_value() ? (*m_variable).max_len(p_window_size) : (*m_fixed).max_len(p_window_size);
	}

	template<typename F>
	void iterate(const SliceStrategy &p_slice_strategy, const F &p_callback) const {

		const size_t n_slices = size();
		size_t token_at = 0;

		for (size_t slice_id = 0;
			slice_id < n_slices;
			slice_id += p_slice_strategy.window_step) {

			const auto len_s = bounded_len(
				slice_id, p_slice_strategy.window_size);

			if (len_s >= 1) {
				if (!p_callback(slice_id, token_at, len_s)) {
					break;
				}
			}

			token_at += bounded_len(
				slice_id, p_slice_strategy.window_step);
		}
	}
};

typedef std::shared_ptr<Spans> SpansRef;


class Booster {
    const py::array_t<float> m_boost_weights;

public:
    Booster(const py::array_t<float> &p_weights) :
        m_boost_weights(p_weights) {
    }

    inline float get_boost(const size_t p_slice_id) const {
        const auto weights = m_boost_weights.unchecked<1>();
        return weights[p_slice_id];
    }
};


class Document :
	public std::enable_shared_from_this<Document>,
	public ContextualVectorsContainer,
	public TokenContainer {

private:
	const int64_t m_id;
	const VocabularyRef m_vocab;

	TokenVectorRef m_tokens;
	size_t m_num_dummy_tokens;
	std::map<std::string, SpansRef> m_spans;

	const py::dict m_metadata;
	std::string m_cache_path;

public:
	Document(
		int64_t p_document_id, // unique index inside corpus
		VocabularyRef p_vocab,
		const py::dict &p_spans,
		const py::dict &p_tokens,
		const py::dict &p_metadata,
		const py::dict &p_contextual_embeddings);

	ResultSetRef find(
	    const QueryRef &p_query,
	    const BoosterRef &p_booster);

	inline VocabularyRef vocabulary() const {
		return m_vocab;
	}

	std::string __str__() const {
		return "<vectorian.core.Document " +
			m_metadata["author"].cast<std::string>() +
			", " +
			m_metadata["title"].cast<std::string>() + ">";
	}

	inline int64_t id() const {
		return m_id;
	}

	const std::string &path() const {
		return m_cache_path;
	}

	const py::dict &metadata() const {
		return m_metadata;
	}

	py::dict py_tokens() const {
		return to_py_array(m_tokens, n_tokens());
	}

	std::string token_str(size_t i) const {
    	return std::string(m_vocab->id_to_token(m_tokens->at(i).id));
	}

	inline const TokenVectorRef &tokens_vector() const {
		return m_tokens;
	}

	inline size_t n_tokens() const {
		return m_tokens->size() - m_num_dummy_tokens;
	}

	virtual std::tuple<const Token*, size_t> tokens() const {
		return std::make_tuple(tokens_vector()->data(), n_tokens());
	}

	const SpansRef &spans(const std::string &p_name) const {
		const auto it = m_spans.find(p_name);
		if (it == m_spans.end()) {
			std::ostringstream err;
			err << "unknown spans " << p_name;
			throw std::runtime_error(err.str());
		}
		return it->second;
	}

	inline size_t max_len(const std::string &p_name, const size_t p_window_size) const {
		return spans(p_name)->max_len(p_window_size);
	}

	py::array_t<int32_t> count_keywords(
   	    const py::dict &p_slice_strategy,
	    const py::tuple &p_keywords) const {

	    std::unordered_set<token_t> keywords;
	    keywords.reserve(p_keywords.size());
	    for (auto x : p_keywords) {
	        const token_t x_id = m_vocab->token_to_id(x.cast<std::string>());
	        if (x_id >= 0) {
    	        keywords.insert(x_id);
	        }
	    }

		const SliceStrategy slice_strategy(p_slice_strategy);
	    const SpansRef spans = this->spans(slice_strategy.level);
	    const size_t n_spans = spans->size();

		const Token *s_tokens = this->tokens_vector()->data();

        py::array_t<size_t> counts{static_cast<py::ssize_t>(n_spans)};
    	auto mutable_counts = counts.mutable_unchecked<1>();
        for (size_t i = 0; i < n_spans; i++) {
            mutable_counts[i] = 0;
        }

	    spans->iterate(p_slice_strategy, [s_tokens, &keywords, &mutable_counts] (
			const size_t slice_id, const size_t token_at, const size_t len_s) {

                for (size_t i = 0; i < len_s; i++) {
                    const Token &t = s_tokens[token_at + i];
                    if (keywords.find(t.id) != keywords.end()) {
                        mutable_counts[slice_id] += 1;
                    }
                }

                return true;
	    });

	    return counts;
	}};

typedef std::shared_ptr<Document> DocumentRef;

#endif // __VECTORIAN_DOCUMENT_H__
