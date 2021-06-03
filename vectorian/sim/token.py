import numpy as np

from vectorian.sim.kernel import Kernel

# inspired by https://github.com/explosion/thinc/blob/master/thinc/types.py
try:
	import cupy
	get_array_module = cupy.get_array_module
except ImportError:
	get_array_module = lambda obj: np


class SimilarityOperator:
	def __call__(self, a, b, out):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class LoggingSimilarity(SimilarityOperator):
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


class CosineSimilarity(SimilarityOperator):
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


class ImprovedSqrtCosineSimilarity(SimilarityOperator):
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


class PNormDistance(SimilarityOperator):
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


class DirectionalDistance(SimilarityOperator):
	def __init__(self, dir):
		self._dir = dir

	def __call__(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		np.linalg.multi_dot([d, self._dir.T], out=out)


class ModifiedMetric(SimilarityOperator):
	def __init__(self, source, *operators):
		self._source = source
		self._kernel = Kernel(operators)

	def __call__(self, a, b, out):
		self._source(a, b, out)
		self._kernel(out)

	@property
	def name(self):
		return self._kernel.name(self._source.name)


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


class TokenSimilarity(AbstractTokenSimilarity):
	def __init__(self, embedding, metric: SimilarityOperator):
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
