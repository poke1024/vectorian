#ifndef __VECTORIAN_WRD__
#define __VECTORIAN_WRD__

#include "alignment/transport.h"
#include <chrono>

struct WRDOptions {
	bool normalize_magnitudes;
	EMDBackend emd_backend;
	float extra_mass_penalty;
};

template<typename FlowRef>
struct WRDSolution {
	float score;
	FlowRef flow;
};

template<typename Index>
class WRD {
	xt::xtensor<float, 1> m_mag_s_storage;
	xt::xtensor<float, 1> m_mag_t_storage;
	xt::xtensor<float, 2> m_cost_storage;
	xt::xtensor<float, 3> m_flow_dist_result;

	OptimalTransport m_ot;

	typedef DenseFlowRef<Index> FlowRef;

	template<typename Slice, typename Vector, typename MatrixD, typename Solution>
	inline void call_debug_hook(
		const QueryRef &p_query, const Slice &slice,
		const int len_s, const int len_t,
		const Vector &mag_s, const Vector &mag_t,
		const MatrixD &D, const int64_t microseconds,
		const Solution &solution) {

		py::gil_scoped_acquire acquire;

		py::dict data = p_query->make_py_debug_slice(slice);

		data["mag_s"] = xt::pyarray<float>(mag_s);
		data["mag_t"] = xt::pyarray<float>(mag_t);

		data["D"] = xt::pyarray<float>(D);

		py::dict py_solution;

		py_solution["G"] = xt::pyarray<float>(solution.G);
		py_solution["cost"] = solution.cost;

		py_solution["type"] = solution.type_str();
		data["solution"] = py_solution;

		data["elapsed_microseconds"] = microseconds;

		const auto callback = *p_query->debug_hook();
		callback("alignment/word-rotators-distance/solver", data);
	}

public:
	template<typename Slice>
	WRDSolution<FlowRef> compute(
		const QueryRef &p_query,
		const Slice &slice,
		const FlowFactoryRef<Index> &p_flow_factory,
		const WRDOptions &p_options) {

		slice.assert_has_magnitudes();

		std::chrono::steady_clock::time_point time_begin;
		if (p_query->debug_hook().has_value()) {
			time_begin = std::chrono::steady_clock::now();
		}

		const size_t len_s = slice.len_s();
		const size_t len_t = slice.len_t();
		const size_t problem_size = len_s + len_t;

		PPK_ASSERT(len_s <= m_mag_s_storage.shape(0));
		PPK_ASSERT(len_t <= m_mag_t_storage.shape(0));

		auto mag_t = xt::view(m_mag_t_storage, xt::range(0, problem_size));
		auto mag_s = xt::view(m_mag_s_storage, xt::range(0, problem_size));
		auto distance_matrix = xt::view(
			m_cost_storage, xt::range(0, problem_size), xt::range(0, problem_size));

		mag_t.fill(0.0f);
		mag_s.fill(0.0f);

		for (size_t i = 0; i < len_t; i++) {
			mag_t(i) = slice.magnitude_t(i);
		}

		for (size_t i = 0; i < len_s; i++) {
			mag_s(len_t + i) = slice.magnitude_s(i);
		}

		if (p_options.normalize_magnitudes) {
			mag_t /= xt::sum(mag_t);
			mag_s /= xt::sum(mag_s);
		}

		distance_matrix.fill(1.0f);
		for (size_t t = 0; t < len_t; t++) {
			for (size_t s = 0; s < len_s; s++) {
				distance_matrix(t, len_t + s) = std::max(0.0f, 1.0f - slice.similarity(s, t));
			}
		}

		const auto r = m_ot.emd(mag_t, mag_s, distance_matrix, p_options.extra_mass_penalty);

		if (p_query->debug_hook().has_value()) {
			const std::chrono::steady_clock::time_point time_end =
				std::chrono::steady_clock::now();
			const auto delta_time = std::chrono::duration_cast<std::chrono::microseconds>(
				time_end - time_begin).count();

			call_debug_hook(
				p_query, slice, len_s, len_t, mag_s, mag_t, distance_matrix, delta_time, r);
		}

		if (r.success()) {
			auto flow_dist_by_pos = xt::view(
				m_flow_dist_result,
				xt::range(0, len_t),
				xt::range(0, len_s),
				xt::all());

			for (size_t t = 0; t < len_t; t++) {
				const float max_flow = mag_t[t];

				for (size_t s = 0; s < len_s; s++) {
					flow_dist_by_pos(t, s, 0) = r.G(t, len_t + s) / max_flow; // normalize
					flow_dist_by_pos(t, s, 1) = distance_matrix(t, len_t + s);
				}
			}

			const float score = (xt::sum((1.0f - distance_matrix) * r.G) / xt::sum(r.G))();

			const auto flow = p_flow_factory->create_dense(flow_dist_by_pos);
			return WRDSolution<FlowRef>{score, flow};
		} else {
			return WRDSolution<FlowRef>{0.0f, FlowRef()};
		}
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		const size_t max_problem_size = max_len_s + max_len_t;

		m_mag_s_storage.resize({max_problem_size});
		m_mag_t_storage.resize({max_problem_size});
		m_cost_storage.resize({max_problem_size, max_problem_size});
		m_flow_dist_result.resize({max_len_t, max_len_s, 2});

		m_ot.resize(max_problem_size);
	}
};

#endif // __VECTORIAN_WRD__
