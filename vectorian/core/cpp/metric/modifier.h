class ModifiedSimilarityMatrixFactory : public SimilarityMatrixFactory {
	const py::object m_operator;
	const std::vector<SimilarityMatrixFactoryRef> m_operands;

	const py::str PY_SIMILARITY;
	const py::str PY_MAGNITUDES_S;
	const py::str PY_MAGNITUDES_T;

public:
	ModifiedSimilarityMatrixFactory(
		const py::object &p_operator,
		const std::vector<SimilarityMatrixFactoryRef> &p_operands);

	virtual SimilarityMatrixRef create(
		const EmbeddingType p_embedding_type,
		const DocumentRef &p_document);
};
