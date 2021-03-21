import numpy as np

from vectorian.alignment import AlignmentAlgorithm, WatermanSmithBeyer
from vectorian.index import BruteForceIndex, SentenceEmbeddingIndex


class VectorSpaceMetric:
	def __call__(self, a, b, out):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class CosineSimilarity(VectorSpaceMetric):
	"""
	cosine gives values in [-1, 1], but Vectorian maps [0, 1] to 0%
	to 100% similarity. usually this is ok, as a cosine of 0 already
	implies orthogonality and thus high unsimilarity. if you want to
	map the whole [-1, 1] range to 0% to 100%, you need to use the
	Bias and Scale modifiers from further below in this file.
	"""

	def __call__(self, a, b, out):
		np.linalg.multi_dot([a.normalized, b.normalized.T], out=out)

	@property
	def name(self):
		return "cosine"


class SqrtCosineSimilarity(VectorSpaceMetric):
	def __call__(self, a, b, out):
		'''
		const float num = xt::sum(xt::sqrt(p_s * p_t))();
		const float denom = xt::sum(p_s)() * xt::sum(p_t)();
		return num / denom;
		'''

	@property
	def name(self):
		return "sqrt-cosine"


class ImprovedSqrtCosineSimilarity(VectorSpaceMetric):
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


class PNormDistance(VectorSpaceMetric):
	def __init__(self, p=2):
		self._p = p

	def __call__(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		d = np.sum(np.power(np.abs(d), self._p), axis=-1)
		d = np.power(d, 1 / self._p)
		out[:, :] = d  # distance, not a similarity

	@property
	def name(self):
		return f"p-norm({self._p})"


class EuclideanDistance(PNormDistance):
	def __init__(self, scale=1):
		super().__init__(p=2, scale=scale)


class DirectionalDistance(VectorSpaceMetric):
	def __init__(self, dir):
		self._dir = dir

	def __call__(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		np.linalg.multi_dot([d, self._dir.T], out=out)


class AbstractTokenSimilarity:
	@property
	def is_modifier(self):
		return False


class TokenSimilarityModifier(AbstractTokenSimilarity):
	@property
	def is_modifier(self):
		return True


class UnaryTokenSimilarityModifier(TokenSimilarityModifier):
	def __init__(self, operand):
		self._operand = operand

	def _compute(self, similarity):
		raise NotImplementedError()

	def __call__(self, operands, out):
		data = operands[0]
		out["similarity"][:] = self._compute(data["similarity"])
		for k in data.keys():
			if k != "similarity":
				out[k][:] = data[k]

	@property
	def operands(self):
		return [self._operand]


class TokenSimilarity(AbstractTokenSimilarity):
	def __init__(self, embedding, metric: VectorSpaceMetric):
		self._embedding = embedding
		self._metric = metric

	def to_args(self):
		return {
			'name': self._embedding.name + "-" + self._metric.name,
			'embedding': self._embedding.name,
			'metric': self._metric
		}


class DistanceToSimilarity(UnaryTokenSimilarityModifier):
	def _compute(self, similarity):
		return np.maximum(0, 1 - similarity)

	@property
	def name(self):
		return f'(1 - {self._operand.name})'


class Bias(UnaryTokenSimilarityModifier):
	def __init__(self, operand, bias):
		super().__init__(operand)
		self._bias = bias

	def _compute(self, similarity):
		return similarity + self._bias

	@property
	def name(self):
		return f'({self._operand.name} + {self._bias})'


class Scale(UnaryTokenSimilarityModifier):
	def __init__(self, operand, scale):
		super().__init__(operand)
		self._scale = scale

	def _compute(self, similarity):
		return similarity * self._scale

	@property
	def name(self):
		return f'({self._operand.name} * {self._scale})'


class MixedTokenSimilarity(TokenSimilarityModifier):
	def __init__(self, metrics, weights):
		self._metrics = metrics
		self._weights = weights

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
		total = np.sum(self._weights)
		terms = []
		for m, w in zip(self._metrics, self._weights):
			terms.append(f'{w / total} * {m.name}')
		return '(' + ' + '.join(terms) + ')'


class ExtremumTokenSimilarity(TokenSimilarityModifier):
	def __init__(self, metrics):
		self._metrics = metrics

	@property
	def operands(self):
		return self._metrics

	def __call__(self, operands, out):
		sim = np.array([x["similarity"] for x in operands])

		sel = np.argmax(sim, axis=0)
		out["similarity"][:] = np.choose(
			sel.reshape(-1),
			sim.reshape(sim.shape[0], -1)).reshape(sim[0].shape)

		if "magnitudes" in out:
			weights = np.apply_along_axis(
				lambda arr: np.bincount(arr, minlength=sim.shape[0]), 1, sel)
			mag = np.array([x["magnitudes"] for x in operands])
			out["magnitudes"] = np.average(mag, axis=0, weights=weights.T)

	@property
	def name(self):
		return f'{self._name}({", ".join([x.name for x in self._metrics])})'


class MaximumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmax
	_name = 'maximum'


class MinimumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmin
	_name = 'minimum'


class SentenceSimilarity:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, partition):
		raise NotImplementedError()


class AlignmentSentenceSimilarity(SentenceSimilarity):
	def __init__(self, token_metric: AbstractTokenSimilarity, alignment=None):
		if not isinstance(token_metric, AbstractTokenSimilarity):
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


class TagWeightedSentenceSimilarity(SentenceSimilarity):
	def __init__(self, token_metric: AbstractTokenSimilarity, alignment, **kwargs):
		assert isinstance(token_metric, AbstractTokenSimilarity)

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


class SentenceEmbeddingSimilarity(SentenceSimilarity):
	"""
	example usage:
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer('paraphrase-distilroberta-base-v1')
	metric = SentenceEmbeddingMetric(model.encode)
	"""

	def __init__(self, encoder, metric=None):
		if metric is None:
			metric = CosineSimilarity()
		assert isinstance(metric, VectorSpaceMetric)
		if not isinstance(metric, CosineSimilarity):
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
