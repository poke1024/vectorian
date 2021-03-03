template<typename Index>
class WRD {
	std::vector<Index> m_match;

public:
	template<typename Slice>
	float compute(
		const Slice &slice,
		const size_t len_s,
		const size_t len_t) {

		return 0.0f;
	}

	void resize(
		const size_t max_len_s,
		const size_t max_len_t) {

		m_match.resize(max_len_t);
	}

	inline const std::vector<Index> &match() const {
		return m_match;
	}

	inline std::vector<Index> &match() {
		return m_match;
	}
};
