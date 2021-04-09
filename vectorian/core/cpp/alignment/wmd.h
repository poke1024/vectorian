#ifndef __VECTORIAN_WMD__
#define __VECTORIAN_WMD__

#include "common.h"
#include "match/match.h"
#include "alignment/transport.h"
#include "alignment/bow.h"

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

	struct Weight {
		float flow;
		float distance;
	};

	struct Edge {
		Index source;
		Index target;
		Weight weight;
	};

	typedef typename BOWProblem<Index>::Document Document;

	static constexpr float MAX_SIMILARITY = 1.0f;

	struct Problem : public BOWProblem<Index> {

		typedef typename BOWProblem<Index>::Document Document;

		xt::xtensor<float, 2> m_distance_matrix;
		std::vector<DistanceRef> m_candidates;
		std::vector<Edge> m_tmp_costs[2];
		xt::xtensor<float, 3> m_flow_dist_result;

		OptimalTransport m_ot;

		// the following values contain the size of
		// the problem currently operated on.

		Index m_len_s;
		Index m_len_t;

		auto mutable_distance_matrix() {
			return xt::view(
				m_distance_matrix,
				xt::range(0, this->m_vocabulary_size),
				xt::range(0, this->m_vocabulary_size));
		}

		auto distance_matrix() const {
			return xt::view(
				m_distance_matrix,
				xt::range(0, this->m_vocabulary_size),
				xt::range(0, this->m_vocabulary_size));
		}

		void allocate(
			const size_t max_len_s,
			const size_t max_len_t) {

			PPK_ASSERT(max_len_s > 0);
			PPK_ASSERT(max_len_t > 0);

			BOWProblem<Index>::allocate(max_len_s, max_len_t);

			const size_t size = max_len_s + max_len_t;

			for (int i = 0; i < 2; i++) {
				m_tmp_costs[i].reserve(size * size);
			}

			m_distance_matrix.resize({size, size});
			m_candidates.reserve(size);

			m_ot.resize(size);
			m_flow_dist_result.resize({max_len_t, max_len_s, 2});
		}

		template<typename Similarity>
		inline void compute_distance_matrix(
			const Similarity &sim,
			const bool make_sparse) {

			Document &doc_s = this->m_doc[0];
			Document &doc_t = this->m_doc[1];

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

template<typename Index>
class WMD {
public:
	struct Problem : public AbstractWMD<Index>::Problem {
	};

	Problem m_problem;

public:
	void allocate(
		const size_t max_len_s,
		const size_t max_len_t) {
		m_problem.allocate(max_len_s, max_len_t);
	}

	template<typename Slice, typename BuildBOW, typename Solver>
	WMDSolution<typename Solver::FlowRef> operator()(
		const QueryRef &p_query,
		const Slice &p_slice,
		BuildBOW &p_build_bow,
		const WMDOptions &p_options,
		const Solver &p_solver) {

		if (p_options.symmetric && !p_options.normalize_bow) {
			// the deeper issue here is that optimal costs of two symmetric
			// problems posed with non-nbow vectors are not comparable, as
			// their scales are - well - not normalized.

			throw std::runtime_error(
				"cannot run symmetric mode WMD with bow (needs nbow)");
		}

		if (!m_problem.initialize(
			p_slice, p_build_bow, p_options.normalize_bow)) {

			return WMDSolution<typename Solver::FlowRef>{
				0.0f, typename Solver::FlowRef()};
		}

		m_problem.m_len_s = p_slice.len_s();
		m_problem.m_len_t = p_slice.len_t();

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

#endif // __VECTORIAN_WMD__
