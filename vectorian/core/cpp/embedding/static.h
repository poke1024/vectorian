#ifndef __VECTORIAN_FAST_EMBEDDING_H__
#define __VECTORIAN_FAST_EMBEDDING_H__

#include "common.h"
#include "embedding/embedding.h"
#include "embedding/sim.h"
#include "utils.h"
#include <xtensor/xadapt.hpp>

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

class Needle {
	const QueryRef m_query;
	const TokenVectorRef m_needle;
	TokenIdArray m_needle_vocabulary_token_ids;
	TokenIdArray m_needle_embedding_token_ids;

public:
	Needle(
		const QueryRef &p_query,
		const VocabularyToEmbedding &p_vocabulary_to_embedding);

	const QueryRef &query() const {
		return m_query;
	}

	const size_t size() const {
		return m_needle->size();
	}

	const TokenIdArray &vocabulary_token_ids() const {
		return m_needle_vocabulary_token_ids;
	}

	const TokenIdArray &embedding_token_ids() const {
		return m_needle_embedding_token_ids;
	}
};

class StaticEmbedding : public Embedding {
	//std::vector<std::string> m_tokens;
	std::unordered_map<std::string, token_t> m_tokens;
	WordVectors m_embeddings;
	//std::map<std::string, SimilarityMatrixBuilderRef> m_similarity_measures;

public:
	StaticEmbedding(
		const std::string &p_name,
		py::object p_table,
		py::object p_progress) : Embedding(p_name) {

		const std::shared_ptr<arrow::Table> table = unwrap_table(p_table);

		/*m_tokens.reserve(table->num_rows());
		iterate_strings(table, "token", [this] (size_t i, const std::string &s) {
			PPK_ASSERT(i == m_tokens.size());
			m_tokens.push_back(s);
		});*/

		const ssize_t stage_size[2] = {
			table->num_rows(),
			table->num_columns()
		};

		const size_t update_freq[2] = {
			1LL << 16,
			10
		};

		const auto progress = [&stage_size, &update_freq, &p_progress] (int stage, size_t i) {
			constexpr size_t num_stages = 2;
			if ((i % update_freq[stage]) == 0) {
				const float p = static_cast<float>(i) / stage_size[stage];
				p_progress((stage + p) / num_stages);
			}
		};

		iterate_strings(table, "token", [this, &progress] (size_t i, const std::string &s) {
			m_tokens[s] = static_cast<long>(i);
			progress(0, i);
		});

		/*{
			auto tokens = string_column(table, "token");

			if (!tokens) {
				throw std::runtime_error("failed to find parquet table column 'token'\n");
			}

			n = tokens->length();
			m_tokens.reserve(n);
			for (int i = 0; i < n; i++) {
				m_tokens[tokens->GetString(i)] = i;
			}
		}*/
		//const size_t n = m_tokens.length();

		// note: these "raw" tables were already normalized in preprocessing.

		m_embeddings.unmodified.resize({m_tokens.size(), static_cast<size_t>(table->num_columns() - 1)});
		/*std::cout << "table size: " << table->num_rows() << " x " << table->num_columns() << "\n";
		std::cout << "m_tokens.size(): " << m_tokens.size() << "\n";
		std::cout << std::flush;*/

		PPK_ASSERT(m_tokens.size() == static_cast<size_t>(table->num_rows()));

		try {
			/*printf("loading embedding vectors parquet table.\n");
			fflush(stdout);*/

			for_each_column<arrow::FloatType, float>(table, [this, &progress] (size_t i, auto v, size_t offset) {
				PPK_ASSERT(i > 0 && offset + v.size() <= m_tokens.size());
				xt::view(m_embeddings.unmodified, xt::range(offset, offset + v.size()), i - 1) = v;
				progress(1, i);
			}, 1);
		} catch(...) {
			printf("failed to load embedding vectors parquet table.\n");
			throw;
		}

		m_embeddings.update_normalized();

		std::cout << p_name << ": " << "loaded " << m_tokens.size() << " tokens." << std::endl;

		/*std::ofstream outfile;
		outfile.open("/Users/arbeit/Desktop/embeddings.txt", std::ios_base::app);
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < m_embeddings.unmodified.shape(1); j++) {
				outfile << m_embeddings.unmodified(i, j) << ",";
			}
			outfile << "\n";
			for (int j = 0; j < m_embeddings.unmodified.shape(1); j++) {
				outfile << m_embeddings.normalized(i, j) << ",";
			}
			outfile << "\n";
		}*/

		//m_similarity_measures = create_similarity_measures(p_name, m_embeddings);
	}

	/*virtual const std::vector<std::string> &tokens() const {
		return m_tokens;
	}*/

	inline const WordVectors &embeddings() const {
		return m_embeddings;
	}

	virtual MetricRef create_metric(
		const QueryRef &p_query,
		const WordMetricDef &p_metric,
		const py::dict &p_sent_metric_def,
		const VocabularyToEmbedding &p_vocabulary_to_embedding);

	py::dict py_vectors() const {
		return m_embeddings.to_py();
	}

	ssize_t token_to_id(const std::string &p_token) const {
		const auto i = m_tokens.find(p_token);
		if (i != m_tokens.end()) {
			return i->second;
		} else {
			return -1;
		}
	}

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

	virtual void update_map(
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
	}

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

#endif // __VECTORIAN_FAST_EMBEDDING_H__
