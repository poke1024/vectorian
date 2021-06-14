import numpy as np

from typing import List

from vectorian.sim.kernel import UnaryOperator
from vectorian.embeddings import AbstractVectors
from vectorian.sim.kernel import Kernel

# inspired by https://github.com/explosion/thinc/blob/master/thinc/types.py
try:
	import cupy
	get_array_module = cupy.get_array_module
except ImportError:
	get_array_module = lambda obj: np


class VectorSimilarity:
	"""A strategy to compute a scalar representing similarity from pairs of vectors."""

	def __call__(self, a: AbstractVectors, b: AbstractVectors, out: np.ndarray):
		self.compute(a, b, out)

	def compute(self, a: AbstractVectors, b: AbstractVectors, out: np.ndarray):
		r"""
		Given n pairs of d-dimensional vectors
		\( ( (\vec{a_1}, \vec{b_1}), ..., (\vec{a_n}, \vec{b_n}) ) \),
		computes some scalar similarity \( {sim}\)
		for each corresponding pair of vectors \( (\vec{a_i}, \vec{b_i}) \). The
		resulting similarity for each pair \( i \) is stored in \( {out}_i \).

		A similarity value of 0 is understood to be minimal (no similarity at all),
		whereas a similarity value of 1 is understood to be maximal (e.g. identity).

		Args:
			a (AbstractVectors): an object providing access to \( \vec{a_1}, ..., \vec{a_n} \)
			b (AbstractVectors): an object providing access to \( \vec{b_1}, ..., \vec{b_n} \)
			out: a preallocated numpy array that upon return has
				\( out_i \) = \( {sim}(\vec{a_i}, \vec{b_i}) \) for all i
		"""
		raise NotImplementedError()

	@property
	def name(self) -> str:
		"""name of the similarity strategy, e.g. 'cosine'"""
		raise NotImplementedError()


class LoggingSimilarity(VectorSimilarity):
	def __init__(self, path, base):
		self._path = path
		self._base = base
		self._file = open(self._path, "w")

	def compute(self, a, b, out):
		import json
		self._file.write(json.dumps({
			'a': a.unmodified.tolist(),
			'b': b.unmodified.tolist()
		}, indent=4))
		self._base(a, b, out)


class CosineSimilarity(VectorSimilarity):
	"""Computes cosine similarity, i.e. the cosine of the angle between two vectors"""

	def compute(self, a, b, out):
		"""
		.. note::
		   By definition in `VectorSimilarity`, the scalars in `out` have meaningful values
		   between 0 and 1. Cosine similarity however generates values in the range [-1, 1].
		   As a cosine of 0 already implies orthogonality and thus high unsimilarity, and
		   negative values are clipped to 0 this is usually not an issue. If you want to
		   regard negative cosine similarity values as similar, you need to modify these
		   values by using `ModifiedVectorSimilarity` with operators such as
		   `vectorian.sim.kernel.Bias` and `vectorian.sim.kernel.Scale`.
		"""

		np.linalg.multi_dot([a.normalized, b.normalized.T], out=out)

	@property
	def name(self):
		return "cosine"


class ImprovedSqrtCosineSimilarity(VectorSimilarity):
	"""
	Sohangir, Sahar, and Dingding Wang. “Improved Sqrt-Cosine Similarity Measurement.”
	Journal of Big Data, vol. 4, no. 1, Dec. 2017, p. 25. DOI.org (Crossref), doi:10.1186/s40537-017-0083-6.

	.. note::
	   The original definition of this similarity strategy is based on BOW and assumes non-negative
	   vectors. However, we work with embeddings that usually contain negative components. Therefore,
	   we apply a simple transformation to make all vectors non-negative.
	"""

	@staticmethod
	def _to_non_negative(x):
		t = np.repeat(x, 2, axis=-1)
		t[:, 1::2] = -t[:, 1::2]
		return np.maximum(0, t)

	def compute(self, a, b, out):
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


class PNormDistance(VectorSimilarity):
	"""Computes distance between vectors using a p-norm.

	.. note::
		By computing a distance instead of a similarity, this component breaks the
		definition of `VectorSimilarity`. In order to be used as a similarity,
		it needs to be combined with `vectorian.sim.kernel.DistanceToSimilarity`.
	"""

	def __init__(self, p: float = 2):
		"""
		Args:
			p (float): the p-norm's p, e.g. 2 for Euclidean
		"""

		self._p = p

	def compute(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		d = np.sum(np.power(np.abs(d), self._p), axis=-1)
		d = np.power(d, 1 / self._p)
		out[:, :] = d  # distance, not a similarity

	@property
	def name(self):
		return f"p-norm({self._p})"


class EuclideanDistance(PNormDistance):
	"""Computes distance between vectors using Euclidean Distance"""

	def __init__(self, scale=1):
		super().__init__(p=2, scale=scale)


class DirectionalDistance(VectorSimilarity):
	def __init__(self, dir):
		self._dir = dir

	def compute(self, a, b, out):
		d = a.unmodified[:, np.newaxis] - b.unmodified[np.newaxis, :]
		np.linalg.multi_dot([d, self._dir.T], out=out)


class ModifiedVectorSimilarity(VectorSimilarity):
	"""VectorSimilarity whose output is modified by one or more operators."""

	def __init__(self, source: VectorSimilarity, *operators: List[UnaryOperator]):
		"""
		Args:
			source (VectorSimilarity): the underlying similarity that is going to
				be modified
			operators (List[UnaryOperator]): one or more operators that modify the
				similarity scalar
		"""

		self._source = source
		self._kernel = Kernel(operators)

	def compute(self, a, b, out):
		self._source(a, b, out)
		self._kernel(out)

	@property
	def name(self):
		return self._kernel.name(self._source.name)
