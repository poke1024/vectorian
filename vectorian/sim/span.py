from vectorian.alignment import SpanFlowStrategy, WatermanSmithBeyer
from vectorian.index import BruteForceIndex, PartitionEmbeddingIndex
from vectorian.sim.token import AbstractTokenSimilarity, SimilarityOperator, CosineSimilarity


class SpanSimilarity:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, index):
		raise NotImplementedError()


class SpanFlowSimilarity(SpanSimilarity):
	def __init__(
		self,
		token_sim: AbstractTokenSimilarity,
		flow_strategy: SpanFlowStrategy = None,
		tag_weights: dict = None,
		**kwargs):

		if not isinstance(token_sim, AbstractTokenSimilarity):
			raise TypeError(token_sim)

		if flow_strategy is None:
			flow_strategy = WatermanSmithBeyer()

		if not isinstance(flow_strategy, SpanFlowStrategy):
			raise TypeError(flow_strategy)

		self._token_sim = token_sim
		self._flow_strategy = flow_strategy
		self._tag_weights = tag_weights
		self._options = kwargs

	@property
	def token_similarity(self):
		return self._token_sim

	@property
	def flow_strategy(self):
		return self._flow_strategy

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, index):
		if not self._tag_weights:
			if self._options:
				raise ValueError(f"illegal option(s): {', '.join(self._options.keys())}")

			return {
				'metric': 'alignment-isolated',
				'token_metric': self._token_sim,
				'alignment': self._flow_strategy.to_args(index.partition)
			}
		else:
			return {
				'metric': 'alignment-tag-weighted',
				'token_metric': self._token_sim,
				'alignment': self._flow_strategy.to_args(index.partition),
				'pos_mismatch_penalty': self._options.get('pos_mismatch_penalty', 0),
				'similarity_threshold': self._options.get('similarity_threshold', 0),
				'tag_weights': self._tag_weights
			}


class SpanEmbeddingSimilarity(SpanSimilarity):
	def __init__(self, encoder, metric=None):
		if metric is None:
			metric = CosineSimilarity()
		assert isinstance(metric, SimilarityOperator)
		if not isinstance(metric, CosineSimilarity):
			raise NotImplementedError()
		self._encoder = encoder
		self._metric = metric

	def create_index(self, partition, **kwargs):
		return PartitionEmbeddingIndex(
			partition, self, self._encoder, **kwargs)

	def load_index(self, session, path, **kwargs):
		return PartitionEmbeddingIndex.load(
			session, self, self._encoder, path, **kwargs)

	def to_args(self, index):
		return None

	@property
	def name(self):
		return "PartitionEmbeddingMetric"