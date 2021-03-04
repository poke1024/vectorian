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

// the following EMD functions have been adapted from:
// https://github.com/PythonOT/POT/tree/master/ot/lp

typedef Eigen::Map<Eigen::MatrixXf> MappedMatrixXf;
typedef Eigen::Map<Eigen::VectorXf> MappedVectorXf;

class EMD {
	MappedMatrixXf m_G_storage;
	MappedVectorXf m_alpha_storage;
	MappedVectorXf m_beta_storage;

	MappedMatrixXf G;
	MappedVectorXf alpha;
	MappedVectorXf beta;

public:
	int emd_c(const MappedVectorXf &a, const MappedVectorXf &b, const MappedMatrixXf &M, const size_t max_iter) {
	    const size_t n1 = M.rows();
	    const size_t n2 = M.cols();
	    const size_t nmax = n1 + n2 - 1;

	    G = MappedMatrixXf(&m_G_storage(0, 0), n1, n2);
	    G.setZero();

	    alpha = MappedVectorXf(&m_alpha_storage(0), n1);
	    alpha.setZero();

	    beta = MappedVectorXf(&m_beta_storage(0), n2);
	    beta.setZero();

	    float cost = 0.0f;

		/*return EMD_wrap(
            n1, n2,
            a, b,
            M, G,
            &cost, max_iter);*/

		return 0;
	}

	int emd2(const MappedVectorXf &a, const MappedVectorXf &b, const MappedMatrixXf &M, const size_t max_iter=100000) {
		PPK_ASSERT(a.rows() == M.rows());
		PPK_ASSERT(b.rows() == M.cols());

		return emd_c(a, b, M, max_iter);
	}
};


template<typename Index>
class WRD {
	std::vector<Index> m_match;

	Eigen::VectorXf m_mag_s_storage;
	Eigen::VectorXf m_mag_t_storage;
	Eigen::MatrixXf m_cost_storage;

public:
	template<typename Slice>
	float compute(
		const Slice &slice,
		const size_t len_s,
		const size_t len_t) {

		MappedVectorXf mag_s(
			&m_mag_s_storage(0), len_s);
		MappedVectorXf mag_t(
			&m_mag_t_storage(0), len_t);
		MappedMatrixXf cost(
			&m_cost_storage(0, 0), len_s, len_t);

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

		// Wd=ot.emd2(a,b,M)

		return 0.0f;
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_mag_s_storage.resize(max_len_s);
		m_mag_t_storage.resize(max_len_t);
		m_cost_storage.resize(max_len_s, max_len_t);

		m_match.resize(max_len_t);
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &match() {
		return m_match;
	}
};
