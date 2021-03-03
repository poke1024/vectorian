import numpy as np

from vectorian.alignment import WatermanSmithBeyer
from vectorian.index import BruteForceIndex, SentenceEmbeddingIndex


class VectorSpaceMetric:
	def to_args(self):
		raise NotImplementedError()

	def index(self, vectors, options):
		def find(u, k):
			for i, v in enumerate(vectors):
				self.score(u, v)
		return find


class CosineMetric(VectorSpaceMetric):
	def to_args(self):
		return {
			'metric': 'cosine',
			'options': {}
		}

	def index(self, vectors):
		vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)


class ZhuCosineMetric(VectorSpaceMetric):
	def to_args(self):
		return {
			'metric': 'zhu-cosine',
			'options': {}
		}


class SohangirCosineMetric(VectorSpaceMetric):
	def to_args(self):
		return {
			'metric': 'sohangir-cosine',
			'options': {}
		}


class PNormMetric(VectorSpaceMetric):
	def __init__(self, p=2, scale=1):
		self._p = p
		self._scale = scale

	def to_args(self):
		return {
			'metric': 'p-norm',
			'options': {
				'p': self._p,
				'scale': self._scale
			}
		}


class EuclideanMetric(PNormMetric):
	def __init__(self, scale=1):
		super().__init__(p=2, scale=scale)


class LerpMetric(VectorSpaceMetric):
	def __init__(self, a: VectorSpaceMetric, b: VectorSpaceMetric, t: float):
		self._a = a
		self._b = b
		self._t = t

	def to_args(self):
		return {
			'name': 'lerp',
			'embedding': None,
			'metric': 'lerp',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args(),
				't': self._t
			}
		}


class MinMetric(VectorSpaceMetric):
	def __init__(self, a: VectorSpaceMetric, b: VectorSpaceMetric):
		self._a = a
		self._b = b

	def to_args(self):
		return {
			'name': 'min',
			'embedding': None,
			'metric': 'min',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args()
			}
		}


class MaxMetric(VectorSpaceMetric):
	def __init__(self, a: VectorSpaceMetric, b: VectorSpaceMetric):
		self._a = a
		self._b = b

	def to_args(self):
		return {
			'name': 'max',
			'embedding': None,
			'metric': 'max',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args()
			}
		}


class TokenSimilarityMetric:
	def __init__(self, embedding, metric: VectorSpaceMetric):
		self._embedding = embedding
		self._metric = metric

	def to_args(self):
		args = self._metric.to_args()
		return {
			'name': self._embedding.name + "-" + args['metric'],
			'embedding': self._embedding.name,
			'metric': args['metric'],
			'options': args['options']
		}


class SentenceSimilarityMetric:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, partition):
		raise NotImplementedError()


class AlignmentSentenceMetric(SentenceSimilarityMetric):
	def __init__(self, word_metric: TokenSimilarityMetric, alignment=None):
		assert isinstance(word_metric, TokenSimilarityMetric)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._word_metric = word_metric
		self._alignment = alignment

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, partition):
		return {
			'metric': 'alignment-isolated',
			'word_metric': self._word_metric.to_args(),
			'alignment': self._alignment.to_args(partition)
		}


class TagWeightedSentenceMetric(SentenceSimilarityMetric):
	def __init__(self, word_metric: TokenSimilarityMetric, alignment, **kwargs):
		assert isinstance(word_metric, TokenSimilarityMetric)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._word_metric = word_metric
		self._alignment = alignment

		self._options = kwargs

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, partition):
		return {
			'metric': 'alignment-tag-weighted',
			'word_metric': self._word_metric.to_args(),
			'alignment': self._alignment.to_args(partition),
			'pos_mismatch_penalty': self._options.get('pos_mismatch_penalty', 0),
			'similarity_threshold': self._options.get('similarity_threshold', 0),
			'tag_weights': self._options.get('tag_weights', {})
		}


class SentenceEmbeddingMetric(SentenceSimilarityMetric):
	"""
	example usage:
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer('paraphrase-distilroberta-base-v1')
	metric = SentenceEmbeddingMetric(model.encode)
	"""

	def __init__(self, encoder, metric=None):
		if metric is None:
			metric = CosineMetric()
		assert isinstance(metric, VectorSpaceMetric)
		if not isinstance(metric, CosineMetric):
			raise NotImplementedError()
		self._encoder = encoder
		self._metric = metric

	def create_index(self, partition, **kwargs):
		return SentenceEmbeddingIndex(
			partition, self, self._encoder, **kwargs)

	def load_index(self, session, path, **kwargs):
		return SentenceEmbeddingIndex.load(
			session, self, self._encoder, path, **kwargs)

	def to_args(self, partition):
		return None

	@property
	def name(self):
		return "SentenceEmbeddingMetric"
