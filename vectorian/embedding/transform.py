import sklearn

from .vectors import Vectors, TransformedVectors


class Transform:
	def apply(self, vectors):
		raise NotImplementedError

	@property
	def name(self):
		raise NotImplementedError


class PCACompression(Transform):
	def __init__(self, n_dims):
		self._n_dims = n_dims

	@property
	def dimension(self):
		return self._n_dims

	def apply(self, vectors):
		pca = sklearn.decomposition.PCA(n_components=self._n_dims)
		pca.fit(vectors.unmodified)

		from skl2onnx import convert_sklearn
		from skl2onnx.common.data_types import FloatTensorType

		onx = convert_sklearn(
			pca, initial_types=[
				("input", FloatTensorType([None, vectors.shape[1]]))])

		return TransformedVectors(
			Vectors(pca.transform(vectors.unmodified)),
			onx.SerializeToString())

	@property
	def name(self):
		return f'pca:{self._n_dims}'