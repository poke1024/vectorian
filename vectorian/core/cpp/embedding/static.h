#ifndef __VECTORIAN_FAST_EMBEDDING_H__
#define __VECTORIAN_FAST_EMBEDDING_H__

#include "common.h"
#include "embedding/embedding.h"
//#include "embedding/sim.h"
#include "embedding/vectors.h"
#include "utils.h"
#include <xtensor/xadapt.hpp>

#if 0
class VocabularyToEmbedding {
	std::vector<TokenIdArray> m_vocabulary_to_embedding;

public:
	inline VocabularyToEmbedding() {
		m_vocabulary_to_embedding.reserve(2);
	}

	const std::vector<TokenIdArray> &unpack() const {
		return m_vocabulary_to_embedding;
	}

	template<typename F>
	inline void iterate(const F &f) const {
		size_t offset = 0;
		for (const auto &embedding_token_ids : m_vocabulary_to_embedding) {
			f(embedding_token_ids, offset);
			offset += embedding_token_ids.shape(0);
		}
	}

	inline void append(const std::vector<token_t> &p_mapping) {
		m_vocabulary_to_embedding.push_back(xt::adapt(
			const_cast<token_t*>(p_mapping.data()), {p_mapping.size()}));
		/*std::cout << "adding vocab mapping with size " <<
			p_mapping.size() << ", total size now: " << this->size() << "\n";*/
	}

	inline size_t size() const {
		size_t vocab_size = 0;
		for (const auto &x : m_vocabulary_to_embedding) {
			vocab_size += x.shape(0);
		}
		return vocab_size;
	}
};
#endif

class Needle {
	const QueryRef m_query;
	const TokenVectorRef m_needle;
	TokenIdArray m_token_ids;

public:
	Needle(const QueryRef &p_query);

	inline const QueryRef &query() const {
		return m_query;
	}

	inline const size_t size() const {
		return m_needle->size();
	}

	inline const TokenIdArray &token_ids() const {
		return m_token_ids;
	}
};

class StaticEmbedding : public Embedding {
	WordVectors m_embeddings;

public:
	StaticEmbedding(
		py::object p_embedding_factory,
		py::list p_tokens);

	inline const WordVectors &vectors() const {
		return m_embeddings;
	}

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const std::vector<EmbeddingRef> &p_embeddings);

	py::dict py_vectors() const {
		return m_embeddings.to_py();
	}

	/*ssize_t token_to_id(const std::string &p_token) const {
		const auto i = m_tokens.find(p_token);
		if (i != m_tokens.end()) {
			return i->second;
		} else {
			return -1;
		}
	}*/

	/*float cosine_similarity(const std::string &p_a, const std::string &p_b) const {
		const auto a = m_tokens.find(p_a);
		const auto b = m_tokens.find(p_b);
		if (a != m_tokens.end() && b != m_tokens.end()) {
			return m_embeddings.raw.row(a->second).dot(m_embeddings.raw.row(b->second));
		} else {
			return 0.0f;
		}
	}*/

	/*MatrixXf similarity_matrix(
		const std::string &p_measure,
		TokenIdArray p_s_embedding_ids,
		TokenIdArray p_t_embedding_ids) const {

		auto i = m_similarity_measures.find(p_measure);
		if (i == m_similarity_measures.end()) {
			throw std::runtime_error("unknown similarity measure");
		}

		MatrixXf m;

		i->second->build_matrix(
			m_embeddings, p_s_embedding_ids, p_t_embedding_ids, m);

		return m;
	}*/

	/*virtual void update_map(
		std::vector<token_t> &p_map,
		const std::vector<std::string> &p_tokens,
		const size_t p_offset) const {

		const size_t i0 = p_map.size();
		const size_t i1 = p_tokens.size();
		PPK_ASSERT(i0 <= i1);
		//std::cout << "update_map called on " << this << ": " << i0 << ", " << i1 << "\n";
		if (i0 == i1) {
			return;
		}
		p_map.resize(i1);

		for (size_t i = i0; i < i1; i++) {
			const auto it = m_tokens.find(p_tokens[i]);
			if (it != m_tokens.end()) {
				p_map[i] = it->second;
			} else {
				p_map[i] = -1;
			}
		}
	}*/

	size_t n_tokens() const {
		return m_embeddings.unmodified.shape(0);
	}

	/*py::list measures() const {
		py::list names;
		for (auto i : m_similarity_measures) {
			names.append(py::str(i.first));
		}
		return names;
	}*/
};

typedef std::shared_ptr<StaticEmbedding> StaticEmbeddingRef;

inline const WordVectors &pick_vectors(
	const std::vector<StaticEmbeddingRef> &p_embeddings,
	const size_t p_index,
	size_t &r_index) {

	size_t i = p_index;
	for (const auto &embedding : p_embeddings) {
		const auto &vectors = embedding->vectors();
		const size_t size = vectors.unmodified.shape(0);
		if (i < size) {
			r_index = i;
			return vectors;
		}
		i -= size;
	}

	std::ostringstream err;
	err << "pick_vectors: " << p_index << " > ";
	for (const auto &embedding : p_embeddings) {
		err << embedding->vectors().unmodified.shape(0);
		if (embedding != p_embeddings.back()) {
			err << " + ";
		}
	}
	throw std::runtime_error(err.str());
}

#endif // __VECTORIAN_FAST_EMBEDDING_H__
