#ifndef __VECTORIAN_TRANSPORT_H__
#define __VECTORIAN_TRANSPORT_H__

#if 1

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"
#include "emd_hat.hpp"
#pragma clang diagnostic pop

class OptimalTransport {
	typedef double scalar_t;

	template<typename T>
	class Matrix {
		std::vector<std::vector<T>> m_cached_rows;
		std::vector<std::vector<T>> m_matrix;

	public:
		void allocate(const size_t max_rows, const size_t max_cols) {
			m_matrix.reserve(max_rows);
			for (size_t i = 0; i < max_rows; i++) {
				m_cached_rows.emplace_back(std::vector<T>(max_cols));
			}
		}

		std::vector<std::vector<T>> &configure(
			const size_t num_rows, const size_t num_cols) {

		}
	};

	xt::xtensor<float, 2> m_G_storage;

	emd_hat_gd_metric<scalar_t, WITHOUT_EXTRA_MASS_FLOW> m_emd;

public:
	void resize(const size_t max_n1, const size_t max_n2) {
		m_G_storage.resize({max_n1, max_n2});
	}

	template<typename Matrix>
	struct Solution {
		float cost;
		Matrix G;

		inline bool success() const {
			return true;
		}

		inline const char *type_str() const {
			return "";
		}
	};

	template<typename Vector, typename Matrix>
	inline auto emd(
		const Vector &a,
		const Vector &b,
		const Matrix &M,
		const float extra_mass_penalty=-1.0f) {

		std::vector<std::vector<scalar_t>> C;
		C.reserve(M.shape(0));
		for (size_t i = 0; i < M.shape(0); i++) {
			C.emplace_back(std::vector<scalar_t>());
			C.back().resize(M.shape(1));
			auto &row = C.back();
			for (size_t j = 0; j < M.shape(1); j++) {
				row[j] = M(i, j);
			}
		}

		const std::vector<scalar_t> P(a.begin(), a.end());
		const std::vector<scalar_t> Q(b.begin(), b.end());
		std::vector<std::vector<scalar_t>> flow_tmp(
			P.size(), std::vector<scalar_t>(P.size()));
		const scalar_t emd = m_emd(
			P, Q, C, extra_mass_penalty, &flow_tmp);

		PPK_ASSERT(P.size() <= m_G_storage.shape(0));
		PPK_ASSERT(Q.size() <= m_G_storage.shape(1));
		auto flow = xt::view(
			m_G_storage, xt::range(0, P.size()), xt::range(0, Q.size()));
		size_t i = 0;
		for (const auto &row : flow_tmp) {
			size_t j = 0;
			for (const auto &col : row) {
				flow(i, j) = col;
				j++;
			}
			i++;
		}

		return Solution<decltype(flow)>{
			static_cast<float>(emd), flow};
	}
};

#else // lemon

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder"
#include "lemon/network_simplex_simple.h"
#pragma clang diagnostic pop

class OptimalTransport {
	// the following EMD functions have been adapted from:
	// https://github.com/PythonOT/POT/tree/master/ot/lp
	// EMD_wrap itself is a port of
	// https://github.com/PythonOT/POT/blob/master/ot/lp/EMD_wrapper.cpp

	xt::xtensor<float, 2> m_G_storage;
	xt::xtensor<float, 1> m_alpha_storage;
	xt::xtensor<float, 1> m_beta_storage;

	typedef uint32_t node_id_type;
    typedef lemon::FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(lemon::FullBipartiteDigraph);
    typedef lemon::NetworkSimplexSimple<Digraph, float, float, node_id_type> Simplex;

public:
    typedef Simplex::ProblemType ProblemType;

private:
	template<typename Vector, typename VectorOut, typename MatrixD, typename MatrixG>
	std::tuple<ProblemType, float> EMD_wrap(
		const int n1, const int n2,
		const Vector &X, const Vector &Y,
		const MatrixD &D, MatrixG &G,
        VectorOut &alpha, VectorOut &beta,
        const int maxIter) {

	    int n, m, cur;

	    // Get the number of non zero coordinates for r and c
	    n=0;
	    for (int i=0; i<n1; i++) {
	        const auto val=X(i);
	        if (val>0) {
	            n++;
	        }else if(val<0){
				return std::make_tuple(ProblemType::INFEASIBLE, 0.0f);
			}
	    }
	    m=0;
	    for (int i=0; i<n2; i++) {
	        const auto val=Y(i);
	        if (val>0) {
	            m++;
	        }else if(val<0){
				return std::make_tuple(ProblemType::INFEASIBLE, 0.0f);
			}
	    }

	    // Define the graph

	    std::vector<int> indI(n), indJ(m);
	    std::vector<float> weights1(n), weights2(m);
	    Digraph di(n, m);
	    Simplex net(di, true, n+m, n*m, maxIter);

	    // Set supply and demand, don't account for 0 values (faster)

	    cur=0;
	    for (int i=0; i<n1; i++) {
	        const auto val=X(i);
	        if (val>0) {
	            weights1[ cur ] = val;
	            indI[cur++]=i;
	        }
	    }

	    // Demand is actually negative supply...

	    cur=0;
	    for (int i=0; i<n2; i++) {
	        const auto val=Y(i);
	        if (val>0) {
	            weights2[ cur ] = -val;
	            indJ[cur++]=i;
	        }
	    }


	    net.supplyMap(&weights1[0], n, &weights2[0], m);

	    // Set the cost of each edge
	    for (int i=0; i<n; i++) {
	        for (int j=0; j<m; j++) {
	            const auto val=D(indI[i], indJ[j]);
	            net.setCost(di.arcFromId(i*m+j), val);
	        }
	    }


	    // Solve the problem with the network simplex algorithm

	    int ret=net.run();
	    float cost = 0.0f;
	    if (ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED) {
	        Arc a; di.first(a);
	        for (; a != INVALID; di.next(a)) {
	            const int i = di.source(a);
	            const int j = di.target(a);
	            const auto flow = net.flow(a);
	            cost += flow * D(indI[i], indJ[j-n]);
	            G(indI[i], indJ[j-n]) = flow;
	            alpha(indI[i]) = -net.potential(i);
	            beta(indJ[j-n]) = net.potential(j);
	        }

	    }

	    return std::make_tuple(
	        static_cast<ProblemType>(ret),
	        cost);
	}

public:
	void resize(const size_t max_n1, const size_t max_n2) {
		m_G_storage.resize({max_n1, max_n2});
		m_alpha_storage.resize({max_n1});
		m_beta_storage.resize({max_n2});
	}

	template<typename Matrix>
	struct Solution {
		ProblemType type;
		float cost;
		Matrix G;

		inline bool success() const {
			return type == ProblemType::OPTIMAL || type == ProblemType::MAX_ITER_REACHED;
		}

		inline const char *type_str() const {
			const char *type_str;
			switch (type) {
				case OptimalTransport::ProblemType::OPTIMAL: {
					type_str = "optimal";
				} break;
				case OptimalTransport::ProblemType::MAX_ITER_REACHED: {
					type_str = "max_iter_reached";
				} break;
				case OptimalTransport::ProblemType::INFEASIBLE: {
					type_str = "infeasible";
				} break;
				case OptimalTransport::ProblemType::UNBOUNDED: {
					type_str = "unbounded";
				} break;
				default: {
					type_str = "illegal";
				} break;
			}
			return type_str;
		}
	};

	template<typename Vector, typename Matrix>
	inline auto emd_c(const Vector &a, const Vector &b, const Matrix &M, const size_t max_iter) {
	    const size_t n1 = M.shape(0);
	    const size_t n2 = M.shape(1);
	    //const size_t nmax = n1 + n2 - 1;

		PPK_ASSERT(a.shape(0) == M.shape(0));
		PPK_ASSERT(b.shape(0) == M.shape(1));

	    PPK_ASSERT(n1 <= static_cast<size_t>(m_alpha_storage.shape(0)));
	    PPK_ASSERT(n2 <= static_cast<size_t>(m_beta_storage.shape(0)));

	    auto G = xt::view(m_G_storage, xt::range(0, n1), xt::range(0, n2));
	    auto alpha = xt::view(m_alpha_storage, xt::range(0, n1));
	    auto beta = xt::view(m_beta_storage, xt::range(0, n2));

	    G.fill(0.0f);
	    alpha.fill(0.0f);
	    beta.fill(0.0f);

		const auto r = EMD_wrap(
            n1, n2,
            a, b,
            M, G,
            alpha, beta, max_iter);

        return Solution<decltype(G)>{std::get<0>(r), std::get<1>(r), G};
	}

	template<typename Vector, typename Matrix>
	inline auto emd2(const Vector &a, const Vector &b, const Matrix &M, const size_t max_iter=100000) {
		return emd_c(a, b, M, max_iter);
	}
};

#endif

#endif // __VECTORIAN_TRANSPORT_H__
