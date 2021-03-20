#include "common.h"
#include "match/match.h"
#include "alignment/transport.h"

#include <xtensor/xadapt.hpp>

struct WMDOptions {
	bool relaxed;
	bool normalize_bow;
	bool symmetric;
	bool injective;
	EMDBackend emd_backend;
	float extra_mass_penalty;
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

	struct Weight {
		float flow;
		float distance;
	};

	struct Edge {
		Index source;
		Index target;
		Weight weight;
	};

	static constexpr float MAX_SIMILARITY = 1.0f;

	struct Problem {
		Document m_doc[2]; // s, t

		xt::xtensor<float, 2> m_distance_matrix;
		std::vector<DistanceRef> m_candidates;
		std::vector<Edge> m_tmp_costs[2];
		xt::xtensor<float, 3> m_flow_dist_result;

		size_t m_max_size; // i.e. pre-allocation

		OptimalTransport m_ot;

		// the following values contain the size of
		// the problem currently operated on.

		Index m_vocabulary_size;
		Index m_len_s;
		Index m_len_t;

		auto mutable_distance_matrix() {
			return xt::view(
				m_distance_matrix,
				xt::range(0, m_vocabulary_size),
				xt::range(0, m_vocabulary_size));
		}

		auto distance_matrix() const {
			return xt::view(
				m_distance_matrix,
				xt::range(0, m_vocabulary_size),
				xt::range(0, m_vocabulary_size));
		}

		auto bow(const int p_doc) const {
			return xt::adapt(
				m_doc[p_doc].bow.data(),
				{m_vocabulary_size});
		}

		py::dict py_vocab_to_pos(const int p_doc) const {
			py::dict result;
			const auto &doc = m_doc[p_doc];
			for (Index i = 0; i < m_vocabulary_size; i++) {
				const auto &mapping = doc.vocab_to_pos[i];
				if (!mapping.empty()) {
					py::list positions;
					for (auto p : mapping) {
						positions.append(p);
					}
					result[py::int_(i)] = positions;
				}
			}
			return result;
		}

		void allocate(
			const size_t max_len_s,
			const size_t max_len_t) {

			PPK_ASSERT(max_len_s > 0);
			PPK_ASSERT(max_len_t > 0);

			const size_t size = max_len_s + max_len_t;

			m_max_size = size;

			for (int i = 0; i < 2; i++) {
				m_doc[i].allocate(size);
				m_tmp_costs[i].reserve(size * size);
			}

			m_distance_matrix.resize({size, size});
			m_candidates.reserve(size);

			m_ot.resize(size);
			m_flow_dist_result.resize({max_len_t, max_len_s, 2});
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
			const Similarity &sim,
			const bool make_sparse) {

			Document &doc_s = m_doc[0];
			Document &doc_t = m_doc[1];

			auto dist = mutable_distance_matrix();

			if (!make_sparse) {
				dist.fill(MAX_SIMILARITY);
			}

			for (const auto &u : doc_s.vocab) {
				const Index i = doc_s.vocab_to_pos[u].front();
				for (const auto &v : doc_t.vocab) {
					const Index j = doc_t.vocab_to_pos[v].front();
					const float d = std::max(MAX_SIMILARITY - sim(i, j), 0.0f);

					// since distance_matrix stores vocabulary indices and
					// not positions into s and t, index of (x, y) vs (y, x)
					// does not matter here and is symmetric.

					dist(u, v) = d;
					dist(v, u) = d;
				}
			}
		}
	};

public:
	static inline float cost_to_score(const float p_cost, const float p_max_cost) {
		return (p_max_cost - p_cost) / p_max_cost;
	}

	class FullSolver {
		const FlowFactoryRef<Index> m_flow_factory;

		template<typename Slice, typename Matrix, typename ByPos>
		void call_debug_hook(
			const QueryRef &p_query,
			const Slice &p_slice,
			const Problem &p_problem,
			const Matrix &G,
			const float score,
			const ByPos &flow_by_pos,
			const ByPos &dist_by_pos) const {

			py::gil_scoped_acquire acquire;

			py::dict data = p_query->make_py_debug_slice(p_slice);

			data["pos_to_vocab_s"] = xt::pyarray<Index>(
				xt::adapt(p_problem.m_doc[0].pos_to_vocab.data(), {p_problem.m_len_s}));
			data["pos_to_vocab_t"] = xt::pyarray<Index>(
				xt::adapt(p_problem.m_doc[1].pos_to_vocab.data(), {p_problem.m_len_t}));

			data["vocab_to_pos_s"] = p_problem.py_vocab_to_pos(0);
			data["vocab_to_pos_t"] = p_problem.py_vocab_to_pos(1);

			data["bow_s"] = xt::pyarray<float>(p_problem.bow(0));
			data["bow_t"] = xt::pyarray<float>(p_problem.bow(1));

			data["D"] = xt::pyarray<float>(p_problem.distance_matrix());
			data["G"] = xt::pyarray<float>(G);

			data["flow_by_pos"] = xt::pyarray<float>(flow_by_pos);
			data["dist_by_pos"] = xt::pyarray<float>(dist_by_pos);

			data["score"] = score;

			const auto callback = *p_query->debug_hook();
			callback("alignment/word-movers-distance/solver", data);
		}

	public:
		typedef DenseFlowRef<Index> FlowRef;

		inline FullSolver(const FlowFactoryRef<Index> &p_flow_factory) :
			m_flow_factory(p_flow_factory) {
		}

		inline bool allow_sparse_distance_matrix() const {
			return false;
		}

		template<typename Slice>
		WMDSolution<FlowRef> operator()(
			const QueryRef &p_query,
			const Slice &p_slice,
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

			const size_t size = p_problem.m_vocabulary_size;
			const auto distance_matrix = p_problem.distance_matrix();

			const Document &doc_s = p_problem.m_doc[0];
			const Document &doc_t = p_problem.m_doc[1];

			auto w_s = p_problem.bow(0);
			auto w_t = p_problem.bow(1);

			const auto r = p_problem.m_ot.emd(
				w_t, w_s, distance_matrix, p_options.extra_mass_penalty);

			if (r.success()) {
				// now map from vocabulary to pos.

				auto flow_dist_by_pos = xt::view(
					p_problem.m_flow_dist_result,
					xt::range(0, p_problem.m_len_t),
					xt::range(0, p_problem.m_len_s));

				PPK_ASSERT(r.G.shape(0) == size);
				PPK_ASSERT(r.G.shape(1) == size);

				for (const Index i : doc_t.vocab) {
					const auto &tpos = doc_t.vocab_to_pos[i];
					const float max_flow = doc_t.bow[i];

					for (const Index j : doc_s.vocab) {
						const auto &spos = doc_s.vocab_to_pos[j];

						for (Index t : tpos) {
							for (Index s : spos) {
								flow_dist_by_pos(t, s, 0) = r.G(i, j) / max_flow; // normalize
								flow_dist_by_pos(t, s, 1) = distance_matrix(i, j);
							}
						}
					}
				}

				const float score = (xt::sum((1.0f - distance_matrix) * r.G) / xt::sum(r.G))();

				if (p_query->debug_hook().has_value()) {
					call_debug_hook(p_query, p_slice, p_problem, r.G, score,
					xt::view(flow_dist_by_pos, xt::all(), xt::all(), 0),
					xt::view(flow_dist_by_pos, xt::all(), xt::all(), 1));
				}

				const auto flow = m_flow_factory->create_dense(flow_dist_by_pos);
				return WMDSolution<FlowRef>{score, flow};
			} else {
				if (p_query->debug_hook().has_value()) {
					call_debug_hook(p_query, p_slice, p_problem, r.G, 0.0f,
						xt::xtensor<float, 2>(), xt::xtensor<float, 2>());
				}

				return WMDSolution<FlowRef>{0.0f, FlowRef()};
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

		inline bool allow_sparse_distance_matrix() const {
			return true;
		}

		template<typename Slice>
		WMDSolution<FlowRef> operator()(
			const QueryRef &p_query,
			const Slice &p_slice,
			Problem &p_problem,
			const WMDOptions &p_options) const {

			// inspired by https://github.com/src-d/wmd-relax

			PPK_ASSERT(p_options.relaxed);

			Document &doc_s = p_problem.m_doc[0];
			Document &doc_t = p_problem.m_doc[1];

			// flipped docs, so we first compute t -> s!
			const Document * const docs[2] = {&doc_t, &doc_s};

			const auto distance_matrix = p_problem.distance_matrix();

			float cost = 0;
			int tighter = 0;
			for (int c = 0; c < 2; c++) {
				p_problem.m_tmp_costs[c].clear();

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
						acc += w1[i] * d;
						p_problem.m_tmp_costs[c].emplace_back(Edge{i, best_j, Weight{w1[i], d}});

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
								acc += remaining * r.d;
								p_problem.m_tmp_costs[c].emplace_back(Edge{i, target, Weight{remaining, r.d}});
								break;
							} else {
								remaining -= w2[target];
								acc += w2[target] * r.d;
								p_problem.m_tmp_costs[c].emplace_back(Edge{i, target, Weight{w2[target], r.d}});
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
			for (const auto &edge : p_problem.m_tmp_costs[tighter]) {
				const auto &spos = doc_s.vocab_to_pos[tighter == 0 ? edge.target : edge.source];
				const auto &tpos = doc_t.vocab_to_pos[tighter == 0 ? edge.source : edge.target];

				const float normalized_flow = edge.weight.flow / (p_options.normalize_bow ?
					1.0f : docs[tighter]->bow[edge.source]);

				for (Index t : tpos) {
					for (Index s : spos) {
						flow->add(t, s, normalized_flow, edge.weight.distance);
					}
				}
			}

			const float max_cost = p_options.normalize_bow ?
				1.0f : p_slice.max_sum_of_similarities();

			return WMDSolution<FlowRef>{
				cost_to_score(cost, max_cost), flow};
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
		const QueryRef &p_query,
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
			},
			p_solver.allow_sparse_distance_matrix());

		return p_solver(
			p_query,
			p_slice,
			m_problem,
			p_options);
	}
};

