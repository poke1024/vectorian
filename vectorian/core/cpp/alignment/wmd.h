#include "common.h"
#include "match/match.h"

#include <xtensor/xadapt.hpp>

struct WMDOptions {
	bool normalize_bow;
	bool symmetric;
	bool one_target;
};

template<typename Index>
struct RelaxedWMDSolution {
	float score;
	SparseFlowRef<Index> flow;
};

template<typename Index, typename WordId>
class WMD {
public:
	struct RefToken {
		WordId word_id;
		Index i; // index in s or t
		int8_t j; // 0 for s, 1 for t
	};

	struct VocabPair {
		Index vocab;
		Index i;
	};

	struct DistanceRef {
		Index i;
		float d;

		inline bool operator<(const DistanceRef &other) const {
			// inverted, so that heap yields smallest elements.
			return d > other.d;
		}
	};

	typedef std::vector<Index> IndexVector;

	struct Document {
		//Eigen::Array<float, Eigen::Dynamic, 1> w1;
		std::vector<float> bow; // (n)bow
		Index w_sum;
		std::vector<VocabPair> vocab;
		// Eigen::Array<Index, 2, Eigen::Dynamic> pos_to_vocab;
		std::vector<Index> pos_to_vocab;
		std::vector<IndexVector> vocab_to_pos; // 1:n

		void resize(WMD &p_wmd, const size_t p_size) {
			bow.resize(p_size);
			vocab.reserve(p_size);
			pos_to_vocab.resize(p_size);
			vocab_to_pos.reserve(p_size);
			for (size_t i = 0; i < p_size; i++) {
				vocab_to_pos.emplace_back(std::vector<Index>());
				vocab_to_pos.back().reserve(p_size);
			}
		}
	};

#if VECTORIAN_MEMORY_POOLS
	foonathan::memory::memory_pool<> m_pool;
#endif

	struct Edge {
		Index source;
		Index target;
		float cost;
	};

	size_t m_size;

	std::vector<RefToken> m_tokens;
	Document m_doc[2]; // s, t

	xt::xtensor<float, 2> m_distance_matrix;
	std::vector<DistanceRef> m_candidates;
	std::vector<Edge> m_results[2];

#if VECTORIAN_MEMORY_POOLS
	WMD() : m_pool(foonathan::memory::list_node_size<Index>::value, 4_KiB) {
	}
#endif

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		PPK_ASSERT(max_len_s > 0);
		PPK_ASSERT(max_len_t > 0);

		const size_t size = max_len_s + max_len_t;

		m_size = size;
		m_tokens.resize(size);

		for (int i = 0; i < 2; i++) {
			m_doc[i].resize(*this, size);
			m_results[i].reserve(size * size);
		}

		m_distance_matrix.resize({size, size});
		m_candidates.reserve(size);
	}

	inline void reset(const int k) {
		for (int j = 0; j < 2; j++) {
			float *w = m_doc[j].bow.data();
			for (int i = 0; i < k; i++) {
				w[i] = 0.0f;
			}
		}
	}

	template<typename Slice, typename MakeWordId>
	inline int init(
		const Slice &slice,
		const MakeWordId &make_word_id,
		const int len_s, const int len_t,
		const WMDOptions &p_options) {

		int k = 0;
		std::vector<RefToken> &z = m_tokens;

		for (int i = 0; i < len_s; i++) {
			z[k++] = RefToken{
				make_word_id(slice.s(i)), static_cast<Index>(i), 0};
		}
		for (int i = 0; i < len_t; i++) {
			z[k++] = RefToken{
				make_word_id(slice.t(i)), static_cast<Index>(i), 1};
		}

		if (k < 1) {
			return 0;
		}

		std::sort(z.begin(), z.begin() + k, [] (const RefToken &a, const RefToken &b) {
			return a.word_id < b.word_id;
		});

		reset(k);

		for (int i = 0; i < 2; i++) {
			auto &doc = m_doc[i];
			doc.w_sum = 0;
			doc.vocab.clear();
			doc.vocab_to_pos[0].clear();
		}

		auto cur_word_id = m_tokens[0].word_id;
		int vocab = 0;
		int vocab_mask = 0;

		for (int i = 0; i < k; i++) {
			const auto &token = m_tokens[i];
			const auto new_word_id = token.word_id;

			const int j = token.j;
			auto &doc = m_doc[j];

			if (new_word_id != cur_word_id) {
				cur_word_id = new_word_id;
				vocab += 1;
				vocab_mask = 0;
				doc.vocab_to_pos[vocab].clear();
			}

			doc.bow[vocab] += 1.0f;
			doc.w_sum += 1;
			doc.pos_to_vocab[token.i] = vocab;
			doc.vocab_to_pos[vocab].push_back(token.i);

			if ((vocab_mask & (1 << j)) == 0) {
				doc.vocab.emplace_back(
					VocabPair{
						static_cast<Index>(vocab),
						token.i});
			}

			vocab_mask |= 1 << j;
		}

		if (p_options.normalize_bow) {
			for (int c = 0; c < 2; c++) {
				float *w = m_doc[c].bow.data();
				const float s = m_doc[c].w_sum;
				for (const auto &u : m_doc[c].vocab) {
					w[u.vocab] /= s;
				}
			}
		}

		return vocab + 1;
	}

	template<typename Similarity>
	inline void compute_distance_matrix(
		const int len_s, const int len_t,
		const int p_size, const Similarity &sim) {

		Document &doc_s = m_doc[0];
		Document &doc_t = m_doc[1];

		auto dist = xt::view(
			m_distance_matrix, xt::range(0, p_size), xt::range(0, p_size));

#if 0
		// since wmd_relaxed will only access dist entries
		// that are sourced from vocab_s and vocab_t, we do
		// not need to initialize the full matrix, which saves
		// us from quadratic time here.

		dist.fill(1.0f);
#endif

		for (const auto &u : doc_s.vocab) {
			for (const auto &v : doc_t.vocab) {
				const float d = 1.0f - sim(u.i, v.i);
				dist(u.vocab, v.vocab) = d;
				dist(v.vocab, u.vocab) = d;
			}
		}
	}

	struct Weights {
		const float *w;
		float sum;
		const std::vector<VocabPair> *v;
	};

	/*float _full() {
		auto distance_matrix = xt::view(
			m_distance_matrix, xt::range(0, p_size), xt::range(0, p_size));

		for (int c = 0; c < 2; c++) {

			auto &w1 = docs[c]->bow;
			auto &w2 = docs[1 - c]->bow;

			auto xw1 = xt::adapt(w1.data(), {w1.size()});
			auto xw2 = xt::adapt(w2.data(), {w2.size()});

			const auto r = m_ot.emd2(xw1, xw2, distance_matrix);
		}
	}*/

	// inspired by implementation in https://github.com/src-d/wmd-relax
	float _relaxed(
		const Index len_s, const Index len_t,
		const Index size,
		const WMDOptions &p_options,
		const SparseFlowRef<Index> &p_flow) {

		constexpr float max_dist = 1.0f; // assume max dist of 1

		Document &doc_s = m_doc[0];
		Document &doc_t = m_doc[1];
		const Document * const docs[2] = {&doc_t, &doc_s};

		const auto distance_matrix = xt::view(
			m_distance_matrix, xt::range(0, size), xt::range(0, size));

		float cost = 0;
		int tighter = 0;
		for (int c = 0; c < 2; c++) {
			m_results[c].clear();

			const float *w1 = docs[c]->bow.data();
			const float *w2 = docs[1 - c]->bow.data();
			const std::vector<VocabPair> &v1 = docs[c]->vocab;
			const std::vector<VocabPair> &v2 = docs[1 - c]->vocab;

			float acc = 0;
			for (const auto &v1_entry : v1) {
				const Index i = v1_entry.vocab;

				if (p_options.one_target) {
					// 1:1 case

					float best_dist = std::numeric_limits<float>::max();
					Index best_j = -1;

					// find argmin.
					for (const auto &v2_entry : v2) {
						const Index j = v2_entry.vocab;
						const float d = distance_matrix(i, j);
						if (d < best_dist) {
							best_dist = d;
							best_j = j;
						}
					}

					// move w1[i] completely to w2[j].
					const float d = (best_j >= 0 ? best_dist : max_dist);
					const float d_acc = w1[i] * d;
					acc += d_acc;
					m_results[c].emplace_back(Edge{i, best_j, d_acc});

				} else {
					// 1:n case

					float remaining = w1[i];

					auto &candidates = m_candidates;
					candidates.clear();

					for (const auto &v2_entry : v2) {
						const int j = v2_entry.vocab;
						const float d = distance_matrix(i, j);
						candidates.push_back(DistanceRef{
							static_cast<Index>(j),
							d});
					}
					std::make_heap(candidates.begin(), candidates.end());

					while (!candidates.empty()) {
						std::pop_heap(candidates.begin(), candidates.end());
						const auto &r = candidates.back();
						const Index target = r.i;

						if (remaining <= w2[target]) {
							const float d_acc = remaining * r.d;
							acc += d_acc;
							m_results[c].emplace_back(Edge{i, target, d_acc});
							break;
						} else {
							remaining -= w2[target];
							const float d_acc = w2[target] * r.d;
							acc += d_acc;
							m_results[c].emplace_back(Edge{i, target, d_acc});
						}

						candidates.pop_back();
					}

					if (remaining > 0.0f) {
						acc += remaining * max_dist;
					}
				}
			}

			if (!p_options.normalize_bow) {
				acc /= docs[c]->w_sum;
			}

			if (!p_options.symmetric) {
				tighter = 0;
				cost = acc;
				break;
			} else if (acc > cost) {
				tighter = c;
				cost = acc;
			}
		}

		// best == 0 -> w1 is t, best == 1 -> w1 is s
		for (const auto &edge : m_results[tighter]) {
			const auto &spos = doc_s.vocab_to_pos[tighter == 0 ? edge.target : edge.source];
			const auto &tpos = doc_t.vocab_to_pos[tighter == 0 ? edge.source : edge.target];

			const float max_cost = (p_options.normalize_bow ?
				1.0f : docs[tighter]->bow[edge.source]);
			const float score = (max_cost - edge.cost) / max_cost;

			for (Index t : tpos) {
				for (Index s : spos) {
					p_flow->add(t, s, score);
				}
			}
		}

		return cost;
	}

	template<typename Slice>
	inline void print_debug(
		const QueryRef &p_query, const Slice &slice,
		const int len_s, const int len_t, const int vocabulary_size,
		std::ostream &os) {

		/*const QueryVocabularyRef vocab = p_query->vocabulary();

		os << "s: ";
		for (int i = 0; i < len_s; i++) {
			os << vocab->id_to_token(slice.s(i).id) << " [" << slice.s(i).id << "]" <<  " ";
		}
		os << "\n";

		os << "t: ";
		for (int i = 0; i < len_t; i++) {
			os << vocab->id_to_token(slice.t(i).id) << " [" << slice.t(i).id << "]" <<  " ";
		}
		os << "\n";

		os << "vocab s: ";
		for (const auto &u : m_doc[0].vocab) {
			os << vocab->id_to_token(slice.s(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "vocab t: ";
		for (const auto &u : m_doc[1].vocab) {
			os << vocab->id_to_token(slice.t(u.i).id) << " [" << u.vocab << "]" << " ";
		}
		os << "\n";

		os << "w1: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_doc[0].bow[i] << " ";
		}
		os << "\n";

		os << "w2: ";
		for (int i = 0; i < vocabulary_size; i++) {
			os << m_doc[1].bow[i] << " ";
		}
		os << "\n";

		os << "dist: \n";
		for (int i = 0; i < vocabulary_size; i++) {
			for (int j = 0; j < vocabulary_size; j++) {
				os << m_dist[i * vocabulary_size + j] << " ";
			}
			os << "\n";
		}*/
	}

	template<typename Slice, typename MakeWordId>
	RelaxedWMDSolution<Index> relaxed(
		const Slice &p_slice,
		const MakeWordId &make_word_id,
		const WMDOptions &p_options,
		const FlowFactoryRef<Index> &p_flow_factory) {

		const int len_s = p_slice.len_s();
		const int len_t = p_slice.len_t();

		if (p_options.symmetric && !p_options.normalize_bow) {
			throw std::runtime_error(
				"cannot run symmetric mode WMD with bow (needs nbow)");
		}

		const float max_score = p_options.normalize_bow ?
			1.0f : p_slice.max_sum_of_similarities();

		const int vocabulary_size = init(
			p_slice, make_word_id, len_s, len_t, p_options);

		if (vocabulary_size == 0) {
			return RelaxedWMDSolution<Index>{0.0f, SparseFlowRef<Index>()};
		}

		const auto flow = p_flow_factory->create_sparse();
		flow->initialize(len_t);

		compute_distance_matrix(
			len_s, len_t,
			vocabulary_size,
			[&p_slice] (int i, int j) -> float {
				return p_slice.similarity(i, j);
			});

		const float score = max_score - _relaxed(
			len_s, len_t,
			vocabulary_size,
			p_options,
			flow);

		return RelaxedWMDSolution<Index>{score, flow};
	}
};