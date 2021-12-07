from cached_property import cached_property
from pathlib import Path

import sys
import numpy as np
import h5py
import sqlite3
import uuid


class AbstractVectors:
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.close()
		return False

	def _save_transform(self, hf):
		pass

	def save(self, path):
		with h5py.File(path.with_suffix(".h5"), "w") as hf:
			hf.create_dataset("unmodified", data=self.unmodified)
			hf.create_dataset("normalized", data=self.normalized)
			hf.create_dataset("magnitudes", data=self.magnitudes)
			self._save_transform(hf)

	def close(self):
		raise NotImplementedError()

	@property
	def memory_usage(self):
		return sys.getsizeof(self.unmodified)

	@property
	def size(self):
		return self.shape[0]

	@property
	def unmodified(self):
		raise NotImplementedError()

	@property
	def normalized(self):
		raise NotImplementedError()

	@property
	def magnitudes(self):
		raise NotImplementedError()

	def transform(self, vectors):
		return vectors


class Vectors(AbstractVectors):
	def __init__(self, unmodified):
		self._unmodified = unmodified

	def close(self):
		pass  # a no op

	@property
	def shape(self):
		return self._unmodified.shape

	@property
	def unmodified(self):
		return self._unmodified

	@cached_property
	def normalized(self):
		eps = np.finfo(np.float32).eps * 100
		vanishing = self.magnitudes < eps
		old_err_settings = np.seterr(divide='ignore', invalid='ignore')
		data = self._unmodified / self.magnitudes[:, np.newaxis]
		np.seterr(**old_err_settings)
		data[vanishing, :].fill(0)
		np.nan_to_num(data, copy=False, nan=0)
		return data

	@cached_property
	def magnitudes(self):
		data = np.linalg.norm(self._unmodified, axis=1)
		np.nan_to_num(data, copy=False, nan=0)
		return data


class TransformedVectors(AbstractVectors):
	# https://scikit-learn.org/stable/modules/model_persistence.html
	# http://onnx.ai/sklearn-onnx/index.html
	# https://github.com/onnx/sklearn-onnx/blob/master/tests/test_sklearn_pca_converter.py
	# http://onnx.ai/sklearn-onnx/auto_examples/plot_benchmark_pipeline.html

	def __init__(self, vectors, onx_spec):
		self._vectors = vectors

		import onnxruntime as rt
		self._sess = rt.InferenceSession(onx_spec)
		self._onx_spec = onx_spec

	def _save_transform(self, hf):
		hf.create_dataset(
			'transform',
			data=np.frombuffer(self._onx_spec, np.uint8))

	def close(self):
		self._vectors.close()

	@property
	def shape(self):
		return self._vectors.shape

	@property
	def unmodified(self):
		return self._vectors.unmodified

	@property
	def normalized(self):
		return self._vectors.normalized

	@property
	def magnitudes(self):
		return self._vectors.magnitudes

	def transform(self, vectors):
		tfm = self._sess.run(
			None, {'input': vectors.unmodified.astype(np.float32)})[0]
		return Vectors(tfm)


class MaskedVectors(AbstractVectors):
	def __init__(self, vectors, mask):
		self._vectors = vectors
		self._mask = mask

	def close(self):
		return self._vectors.close()

	@property
	def size(self):
		return self.shape[0]

	@property
	def shape(self):
		return self.unmodified.shape

	@cached_property
	def unmodified(self):
		return self._vectors.unmodified[self._mask]

	@cached_property
	def normalized(self):
		return self._vectors.normalized[self._mask]

	@cached_property
	def magnitudes(self):
		return self._vectors.magnitudes[self._mask]

	def transform(self, vectors):
		return self._vectors.transform(vectors)


class StackedVectors(AbstractVectors):
	def __init__(self, sources, indices):
		self._sources = sources
		self._indices = indices

		if not all(isinstance(s, Vectors) for s in sources):
			# does not support TransformedVectors that
			# provide a custom transform() operation
			raise RuntimeError(f"unsupported vector source in {sources}")

	def close(self):
		for x in self._sources:
			x.close()

	@cached_property
	def shape(self):
		return self.unmodified.shape

	@cached_property
	def unmodified(self):
		return np.vstack([
			s.unmodified[i] for s, i in zip(self._sources, self._indices)])

	@cached_property
	def normalized(self):
		return np.vstack([
			s.normalized[i] for s, i in zip(self._sources, self._indices)])

	@cached_property
	def magnitudes(self):
		return [s.magnitudes[i] for s, i in zip(self._sources, self._indices)]


class VectorCache:
	def __init__(self, path, readonly=False):
		self._path = Path(path)
		self._readonly = readonly

		self._conn = sqlite3.connect(self._path / 'cache.db')
		self._conn.execute("create table if not exists cache (key varchar primary key, stem varchar)")

	def _get_stem(self, key):
		cur = self._conn.cursor()
		try:
			cur.execute('select stem from cache where key=?', (key, ))
			r = cur.fetchone()
		finally:
			cur.close()

		if r is None:
			return None
		else:
			return r[0]

	def put(self, key, array):
		if self._readonly:
			return

		stem = self._get_stem(key)
		if stem is not None:
			npy_path = self._path / (stem + ".npy")
			np.save(npy_path, array)
		else:
			stem = uuid.uuid1().hex
			npy_path = self._path / (stem + ".npy")
			assert not npy_path.exists()

			with self._conn:
				self._conn.execute("insert into cache(key, stem) values (?, ?)", (
					key, stem))

				np.save(npy_path, array)

	def get(self, key):
		stem = self._get_stem(key)
		if stem is None:
			return None
		else:
			return np.load(self._path / (stem + ".npy"))


class ExternalMemoryVectors:
	@staticmethod
	def load(path):
		hf = h5py.File(path.with_suffix(".h5"), "r")
		v = ExternalMemoryVectors(hf)

		transform = hf.get("transform")
		if transform:
			return TransformedVectors(
				v, np.array(transform).tobytes())
		else:
			return v

	def __init__(self, hf):
		self._hf = hf

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.close()
		return False

	def close(self):
		self._hf.close()

	@property
	def size(self):
		return self.shape[0]

	@property
	def shape(self):
		return self._hf["unmodified"].shape

	@property
	def unmodified(self):
		return np.array(self._hf["unmodified"])

	@property
	def normalized(self):
		return np.array(self._hf["normalized"])

	@cached_property
	def magnitudes(self):
		return np.array(self._hf["magnitudes"])

	def transform(self, vectors):
		return vectors


class OpenedVectorsCache:
	def __init__(self):
		self._cache = dict()

	def open(self, vectors_ref, expected_size):
		cached = None  #self._cache.get(id(vectors_ref))
		if cached is not None:
			_, v = cached
		else:
			v = vectors_ref.open()
			# self._cache[id(vectors_ref)] = (vectors_ref, v)

		if v.size != expected_size:
			raise RuntimeError(f"matrix size mismatch: {v.size} != {expected_size}")
		return v


class VectorsRef:
	def open(self):
		raise NotImplementedError()

	def save(self, path):
		v = self.open()
		try:
			v.save(path)
		finally:
			v.close()

	def compress(self, n_dims):
		v = self.open()
		try:
			r = ProxyVectorsRef(v.compress(n_dims))
		finally:
			v.close()
		return r


class ProxyVectorsRef(VectorsRef):
	def __init__(self, vectors):
		self._vectors = vectors

	def open(self):
		return self._vectors

	def to_internal_memory(self):
		return self


class ExternalMemoryVectorsRef(VectorsRef):
	def __init__(self, path):
		self._path = path

	def open(self):
		return ExternalMemoryVectors.load(self._path)

	def to_internal_memory(self):
		with self.open() as vectors:
			return ProxyVectorsRef(Vectors(vectors.unmodified))


class MaskedVectorsRef(VectorsRef):
	def __init__(self, vectors, mask):
		self._vectors = vectors
		self._mask = mask

	def open(self):
		return MaskedVectors(
			self._vectors.open(), self._mask)

	def to_internal_memory(self):
		return MaskedVectorsRef(
			self._vectors.to_internal_memory(), self._mask)
