/*
ef emd_c(np.ndarray[double, ndim=1, mode="c"] a, np.ndarray[double, ndim=1, mode="c"]  b, np.ndarray[double, ndim=2, mode="c"]  M, int max_iter):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix
        gamm=emd(a,b,M)
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the metric cost matrix
    - a and b are the sample weights
    .. warning::
        Note that the M matrix needs to be a C-order :py.cls:`numpy.array`
    .. warning::
        The C++ solver discards all samples in the distributions with
        zeros weights. This means that while the primal variable (transport
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero.
    Parameters
    ----------
    a : (ns,) numpy.ndarray, float64
        source histogram
    b : (nt,) numpy.ndarray, float64
        target histogram
    M : (ns,nt) numpy.ndarray, float64
        loss matrix
    max_iter : int
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    Returns
    -------
    gamma: (ns x nt) numpy.ndarray
        Optimal transportation matrix for the given parameters
    """
    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]
    cdef int nmax=n1+n2-1
    cdef int result_code = 0
    cdef int nG=0

    cdef double cost=0
    cdef np.ndarray[double, ndim=1, mode="c"] alpha=np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta=np.zeros(n2)

    cdef np.ndarray[double, ndim=2, mode="c"] G=np.zeros([0, 0])

    cdef np.ndarray[double, ndim=1, mode="c"] Gv=np.zeros(0)
    cdef np.ndarray[long, ndim=1, mode="c"] iG=np.zeros(0,dtype=np.int)
    cdef np.ndarray[long, ndim=1, mode="c"] jG=np.zeros(0,dtype=np.int)

    if not len(a):
        a=np.ones((n1,))/n1

    if not len(b):
        b=np.ones((n2,))/n2

    # init OT matrix
    G=np.zeros([n1, n2])

    # calling the function
    with nogil:
        result_code = EMD_wrap(
            n1, n2,
            <double*> a.data, <double*> b.data,
            <double*> M.data, <double*> G.data,
            <double*> alpha.data, <double*> beta.data,
            <double*> &cost, max_iter)

    return G, cost, alpha, beta, result_code
*/

/*


typedef Matrix<float,1,Dynamic> MatrixType;
typedef Map<MatrixType> MapType;
typedef Map<const MatrixType> MapTypeConst;   // a read-only map
const int n_dims = 5;

MatrixType m1(n_dims), m2(n_dims);
m1.setRandom();
m2.setRandom();
float *p = &m2(0);  // get the address storing the data for m2
MapType m2map(p,m2.size());   // m2map shares data with m2
MapTypeConst m2mapconst(p,m2.size());  // a read-only accessor for m2

*/

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder"
#include "network_simplex_simple.h"
#pragma clang diagnostic pop

class OptimalTransport {
	// the following EMD functions have been adapted from:
	// https://github.com/PythonOT/POT/tree/master/ot/lp
	// EMD_wrap itself is a port of
	// https://github.com/PythonOT/POT/blob/master/ot/lp/EMD_wrapper.cpp

	Eigen::MatrixXf m_G_storage;
	Eigen::VectorXf m_alpha_storage;
	Eigen::VectorXf m_beta_storage;

	/*enum ProblemType {
		INFEASIBLE,
		OPTIMAL,
		UNBOUNDED,
		MAX_ITER_REACHED
	};*/

	typedef unsigned int node_id_type;
    typedef lemon::FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(lemon::FullBipartiteDigraph);

	template<typename Vector, typename Matrix>
	std::tuple<bool, float> EMD_wrap(
		const int n1, const int n2,
		const Vector &X, const Vector &Y,
		const Matrix &D, Matrix &G,
        Vector &alpha, Vector &beta,
        const int maxIter) {

	    int n, m, cur;

	    // Get the number of non zero coordinates for r and c
	    n=0;
	    for (int i=0; i<n1; i++) {
	        const auto val=X(i);
	        if (val>0) {
	            n++;
	        }else if(val<0){
				return std::make_tuple(false, 0.0f); //INFEASIBLE;
			}
	    }
	    m=0;
	    for (int i=0; i<n2; i++) {
	        const auto val=Y(i);
	        if (val>0) {
	            m++;
	        }else if(val<0){
				return std::make_tuple(false, 0.0f); //INFEASIBLE;
			}
	    }

	    // Define the graph

	    std::vector<int> indI(n), indJ(m);
	    std::vector<float> weights1(n), weights2(m);
	    Digraph di(n, m);
	    lemon::NetworkSimplexSimple<Digraph,float,float,node_id_type> net(di, true, n+m, n*m, maxIter);

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
	        ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED,
	        cost);
	}

public:
	void resize(const size_t max_n1, const size_t max_n2) {
		m_G_storage.resize(max_n1, max_n2);
		m_alpha_storage.resize(max_n1);
		m_beta_storage.resize(max_n2);
	}

	struct Result {
		bool success;
		float opt_cost;
		//MappedMatrixXf G;
		Eigen::MatrixXf G; // FIXME
	};

	template<typename Vector, typename Matrix>
	inline Result emd_c(const Vector &a, const Vector &b, const Matrix &M, const size_t max_iter) {
	    const size_t n1 = M.rows();
	    const size_t n2 = M.cols();
	    //const size_t nmax = n1 + n2 - 1;

		PPK_ASSERT(a.rows() == M.rows());
		PPK_ASSERT(b.rows() == M.cols());

	    PPK_ASSERT(n1 <= static_cast<size_t>(m_alpha_storage.rows()));
	    PPK_ASSERT(n2 <= static_cast<size_t>(m_beta_storage.rows()));

	    /*auto G = MappedMatrixXf(&m_G_storage(0, 0), n1, n2);
	    G.setZero();

	    auto alpha = MappedVectorXf(&m_alpha_storage(0), n1);
	    alpha.setZero();

	    auto beta = MappedVectorXf(&m_beta_storage(0), n2);
	    beta.setZero();*/

	    MatrixXf G;
	    G.resize(n1, n2);
	    VectorXf alpha;
	    alpha.resize(n1);
	    VectorXf beta;
	    beta.resize(n2); // FIXME

	    G.setZero();
	    alpha.setZero();
	    beta.setZero();

		const auto r = EMD_wrap(
            n1, n2,
            a, b,
            M, G,
            alpha, beta, max_iter);

        return Result{std::get<0>(r), std::get<1>(r), G};
	}

	template<typename Vector, typename Matrix>
	inline Result emd2(const Vector &a, const Vector &b, const Matrix &M, const size_t max_iter=100000) {
		return emd_c(a, b, M, max_iter);
	}
};


template<typename Index>
class WRD {
	std::vector<Index> m_match;

	Eigen::VectorXf m_mag_s_storage;
	Eigen::VectorXf m_mag_t_storage;
	Eigen::MatrixXf m_cost_storage;
	OptimalTransport m_ot;

	template<typename Slice, typename Vector, typename Matrix>
	inline void call_debug_hook(
		const QueryRef &p_query, const Slice &slice,
		const int len_s, const int len_t,
		const Vector &mag_s, const Vector &mag_t,
		const Matrix &D, const Matrix &G) {

		py::gil_scoped_acquire acquire;

		const QueryVocabularyRef vocab = p_query->vocabulary();

		const auto token_vector = [&] (const auto &get_id, const int n) {
			py::list tokens;
			for (int i = 0; i < n; i++) {
				tokens.append(vocab->id_to_token(get_id(i)));
			}
			return tokens;
		};

		py::dict data;

		data[py::str("s")] = token_vector([&] (int i) {
			return slice.s(i).id;
		}, len_s);

		data[py::str("t")] = token_vector([&] (int i) {
			return slice.t(i).id;
		}, len_t);

		data[py::str("mag_s")] = to_py_array(mag_s);
		data[py::str("mag_t")] = to_py_array(mag_t);

		data[py::str("D")] = to_py_array(D);
		data[py::str("G")] = to_py_array(G);

		const auto callback = *p_query->debug_hook();
		callback(data);

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
		const Slice &slice,
		const size_t len_s,
		const size_t len_t) {

		PPK_ASSERT(len_s <= static_cast<size_t>(m_cost_storage.rows()));
		PPK_ASSERT(len_t <= static_cast<size_t>(m_cost_storage.cols()));

		/*MappedVectorXf mag_s(
			&m_mag_s_storage(0), len_s);
		MappedVectorXf mag_t(
			&m_mag_t_storage(0), len_t);
		MappedMatrixXf cost(
			&m_cost_storage(0, 0), len_s, len_t);*/

		VectorXf mag_s;
		mag_s.resize(len_s);
		VectorXf mag_t;
		mag_t.resize(len_t);
		MatrixXf cost;
		cost.resize(len_s, len_t); // FIXME

		for (size_t i = 0; i < len_s; i++) {
			mag_s(i) = slice.magnitude_s(i);
		}
		mag_s /= mag_s.sum();

		for (size_t i = 0; i < len_t; i++) {
			mag_t(i) = slice.magnitude_t(i);
		}
		mag_t /= mag_t.sum();

		for (size_t i = 0; i < len_s; i++) {
			for (size_t j = 0; j < len_t; j++) {
				cost(i, j) = 1.0f - slice.similarity(i, j);
			}
		}

		const auto r = m_ot.emd2(mag_s, mag_t, cost);

		if (r.success) {
			if (p_query->debug_hook().has_value()) {
				call_debug_hook(p_query, slice, len_s, len_t, mag_s, mag_t, cost, r.G);
			}

			return 1.0f - r.opt_cost * 0.5f;
		} else {
			return 0.0f;
		}
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_mag_s_storage.resize(max_len_s);
		m_mag_t_storage.resize(max_len_t);
		m_cost_storage.resize(max_len_s, max_len_t);
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
