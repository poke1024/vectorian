#include "common.h"
#include "match/match.h"
#include "alignment/transport.h"

#include <xtensor/xadapt.hpp>

struct WMDOptions {
	bool relaxed;
	bool normalize_bow;
	bool symmetric;
	bool injective;
};

template<typename FlowRef>
struct WMDSolution {
	float score;
	FlowRef flow;
};

template<typename Index>
class AbstractWMD {
public:
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
		std::vector<float> bow; // (n)bow
		Index w_sum;
		std::vector<Index> vocab;
		std::vector<Index> pos_to_vocab;
		std::vector<IndexVector> vocab_to_pos; // 1:n

		void allocate(const size_t p_size) {
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

	struct Edge {
		Index source;
		Index target;
		float cost;
	};

	static constexpr float MAX_SIMILARITY = 1.0f;

	struct Problem {
		Document m_doc[2]; // s, t

		xt::xtensor<float, 2> m_distance_matrix;
		std::vector<DistanceRef> m_candidates;
		std::vector<Edge> m_results[2];

		size_t m_max_size; // i.e. pre-allocation

		OptimalTransport m_ot;

		// the following values contain the size of
		// the problem currently operated on.

		Index m_vocabulary_size;
		Index m_len_s;
		Index m_len_t;

		void allocate(
			const size_t max_len_s,
			const size_t max_len_t) {

			PPK_ASSERT(max_len_s > 0);
			PPK_ASSERT(max_len_t > 0);

			const size_t size = max_len_s + max_len_t;

			m_max_size = size;

			for (int i = 0; i < 2; i++) {
				m_doc[i].allocate(size);
				m_results[i].reserve(size * size);
			}

			m_distance_matrix.resize({size, size});
			m_candidates.reserve(size);

			m_ot.resize(max_len_s, max_len_t);
		}

		inline void reset(const int k) {
			for (int j = 0; j < 2; j++) {
				float *w = m_doc[j].bow.data();
				for (int i = 0; i < k; i++) {
					w[i] = 0.0f;
				}
			}
		}

		template<typename Similarity>
		inline void compute_distance_matrix(
			const Similarity &sim) {

			Document &doc_s = m_doc[0];
			Document &doc_t = m_doc[1];

			const int size = m_vocabulary_size;

			auto dist = xt::view(
				m_distance_matrix, xt::range(0, size), xt::range(0, size));

#if 0
			// since wmd_relaxed will only access dist entries
			// that are sourced from vocab_s and vocab_t, we do
			// not need to initialize the full matrix, which saves
			// us from quadratic time here.

			dist.fill(MAX_SIMILARITY);
#endif

			for (const auto &u : doc_s.vocab) {
				const Index i = doc_s.vocab_to_pos[u].front();
				for (const auto &v : doc_t.vocab) {
					const Index j = doc_t.vocab_to_pos[v].front();
					const float d = MAX_SIMILARITY - sim(i, j);
					dist(u, v) = d;
					dist(v, u) = d;
				}
			}
		}
	};

public:
	/*struct Weights {
		const float *w;
		float sum;
		const std::vector<Index> *v;
	};*/

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

	template<typename FlowRef>
	struct OptimalCost {
		float cost;
		FlowRef flow;
	};

	class FullSolver {
		const FlowFactoryRef<Index> m_flow_factory;

	public:
		typedef DenseFlowRef<Index> FlowRef;

		inline FullSolver(const FlowFactoryRef<Index> &p_flow_factory) :
			m_flow_factory(p_flow_factory) {
		}

		OptimalCost<FlowRef> operator()(
			Problem &p_problem,
			const WMDOptions &p_options) const {

			PPK_ASSERT(!p_options.relaxed);

			if (p_options.injective) {
				throw std::runtime_error(
					"non-relaxed WMD with injective mapping is not supported");
			}

			if (p_options.symmetric) {
				throw std::runtime_error(
					"non-relaxed WMD with symmetric computation is not supported");
			}

			const Index size = p_problem.m_vocabulary_size;
			const auto distance_matrix = xt::view(
				p_problem.m_distance_matrix,
				xt::range(0, size), xt::range(0, size));

			auto &w1 = p_problem.m_doc[0].bow;
			auto &w2 = p_problem.m_doc[1].bow;

			auto xw1 = xt::adapt(w1.data(), {w1.size()});
			auto xw2 = xt::adapt(w2.data(), {w2.size()});

			const auto r = p_problem.m_ot.emd2(xw1, xw2, distance_matrix);

			if (r.success()) {
				const auto flow = m_flow_factory->create_dense(r.G);
				return OptimalCost<FlowRef>{r.opt_cost, flow};
			} else {
				return OptimalCost<FlowRef>{0.0f, FlowRef()};
			}
		}
	};

	class RelaxedSolver {
		const FlowFactoryRef<Index> m_flow_factory;

	public:
		typedef SparseFlowRef<Index> FlowRef;

		inline RelaxedSolver(const FlowFactoryRef<Index> &p_flow_factory) :
			m_flow_factory(p_flow_factory) {
		}

		OptimalCost<FlowRef> operator()(
			Problem &p_problem,
			const WMDOptions &p_options) const {

			// inspired by https://github.com/src-d/wmd-relax

			PPK_ASSERT(p_options.relaxed);

			Document &doc_s = p_problem.m_doc[0];
			Document &doc_t = p_problem.m_doc[1];
			const Document * const docs[2] = {&doc_t, &doc_s};

			const Index size = p_problem.m_vocabulary_size;
			const auto distance_matrix = xt::view(
				p_problem.m_distance_matrix,
				xt::range(0, size), xt::range(0, size));

			float cost = 0;
			int tighter = 0;
			for (int c = 0; c < 2; c++) {
				p_problem.m_results[c].clear();

				const float *w1 = docs[c]->bow.data();
				const float *w2 = docs[1 - c]->bow.data();
				const std::vector<Index> &v1 = docs[c]->vocab;
				const std::vector<Index> &v2 = docs[1 - c]->vocab;

				float acc = 0;
				for (const Index i : v1) {

					if (p_options.injective) {
						// 1:1 case

						float best_dist = std::numeric_limits<float>::max();
						Index best_j = -1;

						// find argmin.
						for (const Index j : v2) {
							const float d = distance_matrix(i, j);
							if (d < best_dist) {
								best_dist = d;
								best_j = j;
							}
						}

						// move w1[i] completely to w2[j].
						const float d = (best_j >= 0 ? best_dist : MAX_SIMILARITY);
						const float d_acc = w1[i] * d;
						acc += d_acc;
						p_problem.m_results[c].emplace_back(Edge{i, best_j, d_acc});

					} else {
						// 1:n case

						float remaining = w1[i];

						auto &candidates = p_problem.m_candidates;
						candidates.clear();

						for (const Index j : v2) {
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
								p_problem.m_results[c].emplace_back(Edge{i, target, d_acc});
								break;
							} else {
								remaining -= w2[target];
								const float d_acc = w2[target] * r.d;
								acc += d_acc;
								p_problem.m_results[c].emplace_back(Edge{i, target, d_acc});
							}

							candidates.pop_back();
						}

						if (remaining > 0.0f) {
							acc += remaining * MAX_SIMILARITY; // i.e. max distance
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

			const auto flow = m_flow_factory->create_sparse();
			flow->initialize(p_problem.m_len_t);

			// best == 0 -> w1 is t, best == 1 -> w1 is s
			for (const auto &edge : p_problem.m_results[tighter]) {
				const auto &spos = doc_s.vocab_to_pos[tighter == 0 ? edge.target : edge.source];
				const auto &tpos = doc_t.vocab_to_pos[tighter == 0 ? edge.source : edge.target];

				const float max_cost = (p_options.normalize_bow ?
					1.0f : docs[tighter]->bow[edge.source]);
				const float score = (max_cost - edge.cost) / max_cost;

				for (Index t : tpos) {
					for (Index s : spos) {
						flow->add(t, s, score);
					}
				}
			}

			return OptimalCost<FlowRef>{cost, flow};
		}
	};
};

template<typename Index, typename Token>
class WMD {
public:
	struct RefToken {
		Token id; // unique id for token
		Index pos; // index in s or t
		int8_t doc; // 0 for s, 1 for t
	};

	struct Problem : public AbstractWMD<Index>::Problem {
		std::vector<RefToken> m_tokens;

		void allocate(
			const size_t max_len_s,
			const size_t max_len_t) {

			m_tokens.resize(max_len_s + max_len_t);
			AbstractWMD<Index>::Problem::allocate(max_len_s, max_len_t);
		}

		template<typename Slice, typename MakeToken>
		inline bool initialize(
			const Slice &slice,
			const MakeToken &make_token,
			const Index len_s, const Index len_t,
			const WMDOptions &p_options) {

			Index k = 0;
			std::vector<RefToken> &z = m_tokens;

			for (Index i = 0; i < len_s; i++) {
				z[k++] = RefToken{
					make_token(slice.s(i)), i, 0};
			}
			for (Index i = 0; i < len_t; i++) {
				z[k++] = RefToken{
					make_token(slice.t(i)), i, 1};
			}

			if (k < 1) {
				return false;
			}

			std::sort(z.begin(), z.begin() + k, [] (const RefToken &a, const RefToken &b) {
				return a.id < b.id;
			});

			this->reset(k);

			for (int i = 0; i < 2; i++) {
				auto &doc = this->m_doc[i];
				doc.w_sum = 0;
				doc.vocab.clear();
				doc.vocab_to_pos[0].clear();
			}

			auto cur_word_id = m_tokens[0].id;
			Index vocab = 0;

			for (Index i = 0; i < k; i++) {
				const auto &token = m_tokens[i];
				const auto new_word_id = token.id;

				if (new_word_id != cur_word_id) {
					cur_word_id = new_word_id;
					vocab += 1;
					for (int j = 0; j < 2; j++) {
						this->m_doc[j].vocab_to_pos[vocab].clear();
					}
				}

				const int doc_idx = token.doc;
				auto &doc = this->m_doc[doc_idx];

				doc.bow[vocab] += 1.0f;
				doc.w_sum += 1;
				doc.pos_to_vocab[token.pos] = vocab;

				auto &to_pos = doc.vocab_to_pos[vocab];
				if (to_pos.empty()) {
					doc.vocab.push_back(vocab);
				}
				to_pos.push_back(token.pos);
			}

			if (p_options.normalize_bow) {
				for (int c = 0; c < 2; c++) {
					float *w = this->m_doc[c].bow.data();
					const float s = this->m_doc[c].w_sum;
					for (const Index i : this->m_doc[c].vocab) {
						w[i] /= s;
					}
				}
			}

			this->m_vocabulary_size = vocab + 1;
			this->m_len_s = len_s;
			this->m_len_t = len_t;

			return true;
		}
	};

	Problem m_problem;

public:
	void allocate(
		const size_t max_len_s,
		const size_t max_len_t) {
		m_problem.allocate(max_len_s, max_len_t);
	}

	template<typename Slice, typename MakeToken, typename Solver>
	WMDSolution<typename Solver::FlowRef> operator()(
		const Slice &p_slice,
		const MakeToken &p_make_token,
		const WMDOptions &p_options,
		const Solver &p_solver) {

		const auto len_s = p_slice.len_s();
		const auto len_t = p_slice.len_t();

		if (p_options.symmetric && !p_options.normalize_bow) {
			// the deeper issue here is that optimal costs of two symmetric
			// problems posed with non-nbow vectors are not comparable, as
			// their scales are - well - not normalized.

			throw std::runtime_error(
				"cannot run symmetric mode WMD with bow (needs nbow)");
		}

		if (!m_problem.initialize(
			p_slice, p_make_token, len_s, len_t, p_options)) {

			return WMDSolution<typename Solver::FlowRef>{
				0.0f, typename Solver::FlowRef()};
		}

		m_problem.compute_distance_matrix(
			[&p_slice] (int i, int j) -> float {
				return p_slice.similarity(i, j);
			});

		const auto r = p_solver(
			m_problem,
			p_options);

		const float max_cost = p_options.normalize_bow ?
			1.0f : p_slice.max_sum_of_similarities();

		return WMDSolution<typename Solver::FlowRef>{
			max_cost - r.cost, r.flow};
	}
};

