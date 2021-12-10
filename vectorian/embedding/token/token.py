class TokenEmbedding:
	@property
	def is_static(self):
		return False

	@property
	def is_contextual(self):
		return False

	def create_encoder(self):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()

	def to_token_sim(self, vector_sim=None):
		from vectorian.sim.token import EmbeddingTokenSim
		from vectorian.sim.vector import CosineSim

		if vector_sim is None:
			vector_sim = CosineSim()

		return EmbeddingTokenSim(self, vector_sim)

	def to_sentence_sim(self, optimizer, vector_sim=None):
		from vectorian.sim.span import OptimizedSpanSim

		token_sim = self.to_token_sim(vector_sim)
		return OptimizedSpanSim(token_sim, optimizer)

	def to_sentence_embedding(self, agg):
		from vectorian.embedding.span import SentenceEmbedding, AggregatedTokenImpl

		return SentenceEmbedding(AggregatedTokenImpl(self, agg))
