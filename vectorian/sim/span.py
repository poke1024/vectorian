from vectorian.alignment import Optimizer, LocalAlignment
from vectorian.index import BruteForceIndex, SpanEncoderIndex, FaissCosineIndex
from vectorian.sim.token import TokenSim
from vectorian.sim.vector import VectorSim, CosineSim
from vectorian.alignment import ConstantGapCost
from vectorian.embedding.span import SpanEmbedding


class SpanSim:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, index):
		raise NotImplementedError()


class OptimizedSpanSim(SpanSim):
	def __init__(
		self,
		token_sim: TokenSim,
		optimizer: Optimizer = None,
		tag_weights: dict = None,
		**kwargs):

		if not isinstance(token_sim, TokenSim):
			raise TypeError(token_sim)

		if optimizer is None:
			optimizer = LocalAlignment(gap={
				's': ConstantGapCost(0),
				't': ConstantGapCost(0)
			})

		if not isinstance(optimizer, Optimizer):
			raise TypeError(optimizer)

		self._token_sim = token_sim
		self._optimizer = optimizer
		self._tag_weights = tag_weights
		self._options = kwargs

	@property
	def token_sim(self):
		return self._token_sim

	@property
	def optimizer(self):
		return self._optimizer

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, index):
		if not self._tag_weights:
			if self._options:
				raise ValueError(f"illegal option(s): {', '.join(self._options.keys())}")

			return {
				'metric': 'alignment-isolated',
				'token_metric': self._token_sim,
				'alignment': self._optimizer.to_args(index.partition)
			}
		else:
			return {
				'metric': 'alignment-tag-weighted',
				'token_metric': self._token_sim,
				'alignment': self._optimizer.to_args(index.partition),
				'pos_mismatch_penalty': self._options.get('pos_mismatch_penalty', 0),
				'similarity_threshold': self._options.get('similarity_threshold', 0),
				'tag_weights': self._tag_weights
			}


class EmbeddedSpanSim(SpanSim):
	def __init__(self, embedding: SpanEmbedding, sim: VectorSim = None):
		if sim is None:
			sim = CosineSim()
		if not isinstance(sim, VectorSim):
			raise TypeError(f"{sim} is expected to be a VectorSim")
		self._embedding = embedding
		self._vector_sim = sim

	def create_index(self, partition, **kwargs):
		if isinstance(self._vector_sim, CosineSim) and FaissCosineIndex.is_available():
			return FaissCosineIndex(partition, self._embedding, self, **kwargs)
		else:
			return SpanEncoderIndex(
				partition, self._embedding, self, vector_sim=self._vector_sim, **kwargs)

	def to_args(self, index):
		return None

	@property
	def name(self):
		return "EmbeddedSpanSim"
