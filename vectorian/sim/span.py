from vectorian.alignment import Optimizer, LocalAlignment
from vectorian.index import BruteForceIndex, SpanEncoderIndex, FaissCosineIndex
from vectorian.sim.token import TokenSim
from vectorian.sim.vector import VectorSim, CosineSim
from vectorian.alignment import ConstantGapCost
from vectorian.embeddings import TokenEmbedding, SentenceEmbedding


class SpanSim:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, index):
		raise NotImplementedError()

	@staticmethod
	def from_token_embedding(embedding: TokenEmbedding, optimizer: Optimizer, vector_sim=None, **kwargs):
		return TE_SpanSim(embedding.create_token_sim(vector_sim), optimizer, **kwargs)

	@staticmethod
	def from_token_sim(sim: TokenSim, optimizer: Optimizer, **kwargs):
		return TE_SpanSim(sim, optimizer, **kwargs)

	@staticmethod
	def from_sentence_embedding(embedding: SentenceEmbedding = None, encoder=None):
		return SE_SpanSim(embedding)


class TE_SpanSim(SpanSim):  # i.e. SpanSim using TokenEmbeddings
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
	def token_similarity(self):
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


class SE_SpanSim(SpanSim):  # i.e. SpanSim using SentenceEmbeddings
	def __init__(self, embedding: SentenceEmbedding, sim: VectorSim = None):
		if sim is None:
			sim = CosineSim()
		assert isinstance(sim, VectorSim)
		self._encoder = CachedSpanEncoder(embedding)
		self._vector_sim = sim

	def create_index(self, partition, **kwargs):
		if isinstance(self._vector_sim, CosineSim) and FaissCosineIndex.is_available():
			return FaissCosineIndex(partition, self, self._encoder, **kwargs)
		else:
			return SpanEncoderIndex(
				partition, self, self._encoder, vector_sim=self._vector_sim, **kwargs)

	'''
	def load_index(self, session, path, **kwargs):
		return SpanEncoderIndex.load(
			session, self, self._encoder, path, vector_sim=self._vector_sim, **kwargs)
	'''

	def to_args(self, index):
		return None

	@property
	def name(self):
		return "SE_SpanSim"
