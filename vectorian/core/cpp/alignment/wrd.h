#include "alignment/transport.h"

template<typename Index>
class WRD {
	xt::xtensor<float, 1> m_mag_s_storage;
	xt::xtensor<float, 1> m_mag_t_storage;
	xt::xtensor<float, 2> m_cost_storage;
	OptimalTransport m_ot;

	std::vector<Index> m_match;

	template<typename Slice, typename Vector, typename MatrixD, typename Solution>
	inline void call_debug_hook(
		const QueryRef &p_query, const Slice &slice,
		const int len_s, const int len_t,
		const Vector &mag_s, const Vector &mag_t,
		const MatrixD &D, const Solution &solution) {

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

		const auto callback = *p_query->debug_hook();
		callback("alignment/wrd/solver", data);

		/*const auto fmt_matrix = [&] (const MappedMatrixXf &data) {
			fort::char_table table;
			table << "";
			for (int j = 0; j < len_t; j++) {
				table << vocab->id_to_token(slice.t(j).id);
			}
			table << fort::endr;
			for (int i = 0; i < len_s; i++) {
				table << vocab->id_to_token(slice.s(i).id);
				for (int j = 0; j < len_t; j++) {
					table << fmt_float(data(i, j));
				}
				table << fort::endr;
			}
			return table.to_string();
		};*/
	}

public:
	template<typename Slice>
	float compute(
		const QueryRef &p_query,
		const Slice &slice) {

		const size_t len_s = slice.len_s();
		const size_t len_t = slice.len_t();

		PPK_ASSERT(len_s <= static_cast<size_t>(m_cost_storage.shape(0)));
		PPK_ASSERT(len_t <= static_cast<size_t>(m_cost_storage.shape(1)));

		/*MappedVectorXf mag_s(
			&m_mag_s_storage(0), len_s);
		MappedVectorXf mag_t(
			&m_mag_t_storage(0), len_t);
		MappedMatrixXf cost(
			&m_cost_storage(0, 0), len_s, len_t);*/

		xt::xtensor<float, 1> mag_s;
		mag_s.resize({len_s});
		xt::xtensor<float, 1> mag_t;
		mag_t.resize({len_t});
		xt::xtensor<float, 2> cost({len_s, len_t});

		for (size_t i = 0; i < len_s; i++) {
			mag_s(i) = slice.magnitude_s(i);
		}

		for (size_t i = 0; i < len_t; i++) {
			mag_t(i) = slice.magnitude_t(i);
		}

		mag_s /= xt::sum(mag_s);
		mag_t /= xt::sum(mag_t);

		/*std::ofstream outfile;
		outfile.open("/Users/arbeit/Desktop/debug_wrd.txt", std::ios_base::app);

		outfile << "--- before:\n";*/
		for (size_t i = 0; i < len_s; i++) {
			for (size_t j = 0; j < len_t; j++) {
				cost(i, j) = 1.0f - slice.similarity(i, j);
				//outfile << cost(i, j) << "; " << slice.similarity(i, j) << "\n";
			}
		}

		const auto r = m_ot.emd(mag_s, mag_t, cost);

		if (p_query->debug_hook().has_value()) {
			call_debug_hook(p_query, slice, len_s, len_t, mag_s, mag_t, cost, r);
		}

		/*outfile << "--- after:\n";
		for (size_t i = 0; i < len_s; i++) {
			for (size_t j = 0; j < len_t; j++) {
				outfile << cost(i, j) << "\n";
			}
		}*/

		if (r.success()) {
			//outfile << "--- success.\n";
			return 1.0f - r.cost * 0.5f;
		} else {
			return 0.0f;
		}
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_mag_s_storage.resize({max_len_s});
		m_mag_t_storage.resize({max_len_t});
		m_cost_storage.resize({max_len_s, max_len_t});
		m_ot.resize(max_len_s, max_len_t);

		m_match.reserve(max_len_t);
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &match() {
		return m_match;
	}
};
