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
        result_code = EMD_wrap(n1, n2, <double*> a.data, <double*> b.data, <double*> M.data, <double*> G.data, <double*> alpha.data, <double*> beta.data, <double*> &cost, max_iter)

    return G, cost, alpha, beta, result_code
*/


/*

	def emd2(a, b, M, processes=multiprocessing.cpu_count(),
         numItermax=100000, log=False, return_matrix=False,
         center_dual=True):

	a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
        "Dimension mismatch, check dimensions of M with a and b"

    asel = a != 0

    if log or return_matrix:
        def f(b):
            bsel = b != 0

            G, cost, u, v, result_code = emd_c(a, b, M, numItermax)


	if len(b.shape) == 1:
        return f(b)


*/


template<typename Index>
class WRD {
	std::vector<Index> m_match;

	ArrayXf m_mag_s;
	ArrayXf m_mag_t;
	MatrixXf m_cost;

public:
	template<typename Slice>
	float compute(
		const Slice &slice,
		const size_t len_s,
		const size_t len_t) {

		for (size_t i = 0; i < len_s; i++) {
			m_mag_s(i) = slice.magnitude_s(i);
		}
		m_mag_s /= m_mag_s.sum();

		for (size_t i = 0; i < len_t; i++) {
			m_mag_t(i) = slice.magnitude_t(i);
		}
		m_mag_t /= m_mag_t.sum();

		for (size_t i = 0; i < len_s; i++) {
			for (size_t j = 0; j < len_t; j++) {
				m_cost(i, j) = 1.0f - slice.similarity(i, j);
			}
		}

		// Wd=ot.emd2(a,b,M)

		return 0.0f;
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_mag_s.resize(max_len_s);
		m_mag_t.resize(max_len_t);
		m_cost.resize(max_len_s, max_len_t);

		m_match.resize(max_len_t);
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &match() {
		return m_match;
	}
};
