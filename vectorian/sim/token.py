from vectorian.sim.vector import VectorSim


class AbstractTokenSim:
	@property
	def is_modifier(self):
		return False

	@property
	def name(self):
		raise NotImplementedError()

	@property
	def embeddings(self):
		raise NotImplementedError()


class TokenSim(AbstractTokenSim):
	def __init__(self, embedding, sim: VectorSim):
		self._embedding = embedding
		self._sim = sim

	@property
	def name(self):
		return f'{self._sim.name}[{self._embedding.name}]'

	@property
	def embeddings(self):
		return [self._embedding]

	@property
	def embedding(self):
		return self._embedding

	@property
	def similarity(self):
		return self._sim

	def to_args(self, index):
		return {
			'name': self._embedding.name + "-" + self._sim.name,
			'embedding': self._embedding.name,
			'metric': self._sim
		}
