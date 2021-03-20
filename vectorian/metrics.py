import numpy as np

from vectorian.alignment import AlignmentAlgorithm, WatermanSmithBeyer
from vectorian.index import BruteForceIndex, SentenceEmbeddingIndex


class VectorSpaceMetric:
	@property
	def is_interpolator(self):
		return False

	def __call__(self, a, b, out):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class CosineMetric(VectorSpaceMetric):
	def __call__(self, a, b, out):
		np.linalg.multi_dot([a.normalized, b.normalized.T], out=out)

	@property
	def name(self):
		return "cosine"


class SqrtCosineMetric(VectorSpaceMetric):
	def __call__(self, a, b, out):
		'''
		const float num = xt::sum(xt::sqrt(p_s * p_t))();
		const float denom = xt::sum(p_s)() * xt::sum(p_t)();
		return num / denom;
		'''

	@property
	def name(self):
		return "sqrt-cosine"


class ImprovedSqrtCosineMetric(VectorSpaceMetric):
	"""
	Sohangir, Sahar, and Dingding Wang. “Improved Sqrt-Cosine Similarity Measurement.”
	Journal of Big Data, vol. 4, no. 1, Dec. 2017, p. 25. DOI.org (Crossref), doi:10.1186/s40537-017-0083-6.
	"""

	def __call__(self, a, b, out):
		num = np.sum(np.sqrt(a.unmodified[:, np.newaxis] * b.unmodified[np.newaxis, :]), axis=-1)
		x = np.sqrt(np.sum(a.unmodified, axis=-1))
		y = np.sqrt(np.sum(b.unmodified, axis=-1))
		denom = x[:, np.newaxis] * y[np.newaxis, :]
		out[:, :] = num / denom

	@property
	def name(self):
		return "improved-sqrt-cosine"


class PNormMetric(VectorSpaceMetric):
	def __init__(self, p=2, scale=1):
		self._p = p
		self._scale = scale

	def __call__(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		d = np.sum(np.power(np.abs(d), self._p), axis=-1)
		d = np.power(d, 1 / self._p)

		# now convert distance to similarity measure.
		out[:, :] = np.maximum(0, 1 - d * self._scale)

	@property
	def name(self):
		return f"p-norm({self._p})"


class EuclideanMetric(PNormMetric):
	def __init__(self, scale=1):
		super().__init__(p=2, scale=scale)


class AbstractTokenSimilarityMeasure:
	@property
	def is_interpolator(self):
		return False


class TokenSimilarityMeasure(AbstractTokenSimilarityMeasure):
	def __init__(self, embedding, metric: VectorSpaceMetric):
		self._embedding = embedding
		self._metric = metric

	def to_args(self):
		return {
			'name': self._embedding.name + "-" + self._metric.name,
			'embedding': self._embedding.name,
			'metric': self._metric
		}


class MixedTokenSimilarityMeasure(AbstractTokenSimilarityMeasure):
	def __init__(self, metrics, weights):
		self._metrics = metrics
		self._weights = weights

	@property
	def is_interpolator(self):
		return True

	@property
	def operands(self):
		return self._metrics

	def __call__(self, operands, out):
		for k in out.keys():
			data = [x[k] for x in operands]
			avg = np.average(data, axis=0, weights=self._weights)
			assert avg.shape == out[k].shape
			out[k][:] = avg

	@property
	def name(self):
		return "mixed"



class MinMetric(VectorSpaceMetric):
	def __init__(self, a: VectorSpaceMetric, b: VectorSpaceMetric):
		self._a = a
		self._b = b



class MaxMetric(VectorSpaceMetric):
	def __init__(self, a: VectorSpaceMetric, b: VectorSpaceMetric):
		self._a = a
		self._b = b


class SentenceSimilarityMetric:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, partition):
		raise NotImplementedError()


class AlignmentSentenceMetric(SentenceSimilarityMetric):
	def __init__(self, token_metric: AbstractTokenSimilarityMeasure, alignment=None):
		if not isinstance(token_metric, AbstractTokenSimilarityMeasure):
			raise TypeError(token_metric)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		if not isinstance(alignment, AlignmentAlgorithm):
			raise TypeError(alignment)

		self._token_metric = token_metric
		self._alignment = alignment

	@property
	def token_similarity_metric(self):
		return self._token_metric

	@property
	def alignment(self):
		return self._alignment

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, partition):
		return {
			'metric': 'alignment-isolated',
			'token_metric': self._token_metric,
			'alignment': self._alignment.to_args(partition)
		}


class TagWeightedSentenceMetric(SentenceSimilarityMetric):
	def __init__(self, token_metric: AbstractTokenSimilarityMeasure, alignment, **kwargs):
		assert isinstance(token_metric, AbstractTokenSimilarityMeasure)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._token_metric = token_metric
		self._alignment = alignment

		self._options = kwargs

	@property
	def token_similarity_metric(self):
		return self._token_metric

	@property
	def alignment(self):
		return self._alignment

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, partition):
		return {
			'metric': 'alignment-tag-weighted',
			'token_metric': self._token_metric,
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
