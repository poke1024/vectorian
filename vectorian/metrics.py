import numpy as np

from vectorian.alignment import AlignmentAlgorithm, WatermanSmithBeyer
from vectorian.index import BruteForceIndex, PartitionEmbeddingIndex


# inspired by https://github.com/explosion/thinc/blob/master/thinc/types.py
try:
	import cupy
	get_array_module = cupy.get_array_module
except ImportError:
	get_array_module = lambda obj: np


class VectorSpaceMetric:
	def __call__(self, a, b, out):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class LoggingSimilarity(VectorSpaceMetric):
	def __init__(self, path, base):
		self._path = path
		self._base = base
		self._file = open(self._path, "w")

	def __call__(self, a, b, out):
		import json
		self._file.write(json.dumps({
			'a': a.unmodified.tolist(),
			'b': b.unmodified.tolist()
		}, indent=4))
		self._base(a, b, out)


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


class ImprovedSqrtCosineSimilarity(VectorSpaceMetric):
	"""
	Sohangir, Sahar, and Dingding Wang. “Improved Sqrt-Cosine Similarity Measurement.”
	Journal of Big Data, vol. 4, no. 1, Dec. 2017, p. 25. DOI.org (Crossref), doi:10.1186/s40537-017-0083-6.
	"""

	@staticmethod
	def _to_non_negative(x):
		t = np.repeat(x, 2, axis=-1)
		t[:, 1::2] = -t[:, 1::2]
		return np.maximum(0, t)

	def __call__(self, a, b, out):
		# we assume embeddings that may be negative. however this distance
		# assumes non-negative vectors so we apply a simple transformation
		# to make all vectors non-negative.

		a_pos = self._to_non_negative(a.unmodified)
		b_pos = self._to_non_negative(b.unmodified)

		num = np.sum(np.sqrt(a_pos[:, np.newaxis] * b_pos[np.newaxis, :]), axis=-1)

		x = np.sqrt(np.sum(a_pos, axis=-1))
		y = np.sqrt(np.sum(b_pos, axis=-1))
		denom = x[:, np.newaxis] * y[np.newaxis, :]

		old_err_settings = np.seterr(divide='ignore', invalid='ignore')
		out[:, :] = num / denom
		np.seterr(**old_err_settings)
		np.nan_to_num(out, copy=False, nan=0)

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


class UnaryOperator:
	def kernel(self, data):
		raise NotImplementedError()

	def name(self, operand):
		raise NotImplementedError()


class Kernel:
	def __init__(self, operators):
		self._operators = operators

		# might JIT in the future.
		def chain(data):
			for x in operators:
				x.kernel(data)

		self._kernel = chain

	def __call__(self, data):
		self._kernel(data)

	def name(self, operand):
		name = operand
		for x in self._operators:
			name = x.name(name)
		return name


class ModifiedMetric(VectorSpaceMetric):
	def __init__(self, source, *operators):
		self._source = source
		self._kernel = Kernel(operators)

	def __call__(self, a, b, out):
		self._source(a, b, out)
		self._kernel(out)

	@property
	def name(self):
		return self._kernel.name(self._source.name)


class RadialBasis(UnaryOperator):
	def __init__(self, gamma):
		self._gamma = gamma

	def kernel(self, data):
		data[:, :] = np.exp(-self._gamma * np.power(data, 2))

	def name(self, operand):
		return f'radialbasis({operand}, {self._gamma})'


class DistanceToSimilarity(UnaryOperator):
	def kernel(self, data):
		data[:, :] = np.maximum(0, 1 - data)

	def name(self, operand):
		return f'(1 - {operand})'


class Bias(UnaryOperator):
	def __init__(self, bias):
		self._bias = bias

	def kernel(self, data):
		data += self._bias

	def name(self, operand):
		return f'({operand} + {self._bias})'


class Scale(UnaryOperator):
	def __init__(self, scale):
		self._scale = scale

	def kernel(self, data):
		data *= self._scale

	def name(self, operand):
		return f'({operand} * {self._scale})'


class Power(UnaryOperator):
	def __init__(self, exp):
		self._exp = exp

	def kernel(self, data):
		data[:, :] = np.power(np.maximum(data, 0), self._exp)

	def name(self, operand):
		return f'({operand} ** {self._exp})'


class Threshold(UnaryOperator):
	def __init__(self, threshold):
		self._threshold = threshold

	def kernel(self, data):
		x = np.maximum(0, data - self._threshold)
		x[x > 0] += self._threshold
		data[:, :] = x

	def name(self, operand):
		return f'threshold({operand}, {self._threshold})'


class AbstractTokenSimilarity:
	@property
	def is_modifier(self):
		return False

	@property
	def name(self):
		raise NotImplementedError()

	@property
	def embeddings(self):
		raise NotImplementedError()


class TokenSimilarityModifier(AbstractTokenSimilarity):
	@property
	def is_modifier(self):
		return True

	@property
	def name(self):
		raise NotImplementedError()

	@property
	def embeddings(self):
		raise NotImplementedError()


class UnaryTokenSimilarityModifier(TokenSimilarityModifier):
	def __init__(self, operand, operators):
		self._operand = operand
		self._kernel = Kernel(operators)

	def __call__(self, operands, out):
		data = operands[0]

		out["similarity"][:] = data["similarity"]
		self._kernel(out["similarity"])

		for k in data.keys():
			if k != "similarity":
				out[k][:] = data[k]

	@property
	def operands(self):
		return [self._operand]

	@property
	def name(self):
		return self._kernel.name(self._operand.name)

	@property
	def embeddings(self):
		return [x.embedding for x in self._operand]


class TokenSimilarity(AbstractTokenSimilarity):
	def __init__(self, embedding, metric: VectorSpaceMetric):
		self._embedding = embedding
		self._metric = metric

	@property
	def name(self):
		return f'{self._metric.name}[{self._embedding.name}]'

	@property
	def embeddings(self):
		return [self._embedding]

	@property
	def embedding(self):
		return self._embedding

	@property
	def metric(self):
		return self._metric

	def to_args(self, index):
		return {
			'name': self._embedding.name + "-" + self._metric.name,
			'embedding': self._embedding.name,
			'metric': self._metric
		}


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
		return f'({" + ".join(terms)})'

	@property
	def embeddings(self):
		return [x.embedding for x in self._metrics]


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

	@property
	def embeddings(self):
		return [x.embedding for x in self._metrics]


class MaximumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmax
	_name = 'maximum'


class MinimumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmin
	_name = 'minimum'


class SpanSimilarity:
	def create_index(self, partition):
		raise NotImplementedError()

	def to_args(self, index):
		raise NotImplementedError()


class AlignmentSimilarity(SpanSimilarity):
	def __init__(self, token_sim: AbstractTokenSimilarity, alignment=None):
		if not isinstance(token_sim, AbstractTokenSimilarity):
			raise TypeError(token_sim)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		if not isinstance(alignment, AlignmentAlgorithm):
			raise TypeError(alignment)

		self._token_sim = token_sim
		self._alignment = alignment

	@property
	def token_similarity(self):
		return self._token_sim

	@property
	def alignment(self):
		return self._alignment

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, index):
		return {
			'metric': 'alignment-isolated',
			'token_metric': self._token_sim,
			'alignment': self._alignment.to_args(index.partition)
		}


class TagWeightedSimilarity(SpanSimilarity):
	def __init__(self, token_sim: AbstractTokenSimilarity, alignment, **kwargs):
		assert isinstance(token_sim, AbstractTokenSimilarity)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._token_sim = token_sim
		self._alignment = alignment

		self._options = kwargs

	@property
	def token_similarity(self):
		return self._token_sim

	@property
	def alignment(self):
		return self._alignment

	def create_index(self, partition, **kwargs):
		return BruteForceIndex(partition, self, **kwargs)

	def to_args(self, index):
		return {
			'metric': 'alignment-tag-weighted',
			'token_metric': self._token_sim,
			'alignment': self._alignment.to_args(index.partition),
			'pos_mismatch_penalty': self._options.get('pos_mismatch_penalty', 0),
			'similarity_threshold': self._options.get('similarity_threshold', 0),
			'tag_weights': self._options.get('tag_weights', {})
		}


class SpanEmbeddingSimilarity(SpanSimilarity):
	def __init__(self, encoder, metric=None):
		if metric is None:
			metric = CosineSimilarity()
		assert isinstance(metric, VectorSpaceMetric)
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
