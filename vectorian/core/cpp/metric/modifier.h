class ModifiedMetricFactory {
	const py::object m_operator;
	const std::vector<MetricRef> m_operands;

public:
	ModifiedMetricFactory(
		const py::object &p_operator,
		const std::vector<MetricRef> &p_operands) :

		m_operator(p_operator),
		m_operands(p_operands) {
	}

	MetricRef create(
		const QueryRef &p_query);
};
