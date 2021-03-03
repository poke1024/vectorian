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
