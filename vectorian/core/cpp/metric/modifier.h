class ModifiedSimilarityMatrixFactory : public SimilarityMatrixFactory {
	const py::object m_operator;
	const std::vector<SimilarityMatrixFactoryRef> m_operands;

public:
	ModifiedSimilarityMatrixFactory(
		const py::object &p_operator,
		const std::vector<SimilarityMatrixFactoryRef> &p_operands) :

		m_operator(p_operator),
		m_operands(p_operands) {
	}

	virtual SimilarityMatrixRef create(
		const EmbeddingType p_embedding_type,
		const DocumentRef &p_document);
};
