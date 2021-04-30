import vectorian.core as core

from tqdm import tqdm
from pathlib import Path
from cached_property import cached_property

import collections
import numpy as np
import json
import os
import sys
import compress_fasttext
import h5py
import sklearn
import time
import contextlib
import io
import re
import gensim
import gensim.models
import requests
import zipfile
import urllib.parse
import cachetools


_gensim_version = int(gensim.__version__.split(".")[0])
if _gensim_version >= 4:
	raise RuntimeError("Vectorian needs gensim < 4.0.0")


_custom_cache_path = None


def set_cache_path(path):
	global _custom_cache_path
	_custom_cache_path = Path(path)


def _make_cache_path():
	if _custom_cache_path is None:
		cache_path = Path.home() / ".vectorian" / "embeddings"
	else:
		cache_path = _custom_cache_path / "embeddings"
	cache_path.mkdir(exist_ok=True, parents=True)
	return cache_path


def _download(url, path, force_download=False):
	path = Path(path)
	download_path = path / urllib.parse.urlparse(url).path.split("/")[-1]
	is_zip = download_path.suffix == ".zip"

	if is_zip:
		result_path = path / download_path.stem
	else:
		result_path = download_path

	if result_path.exists() and not force_download:
		return result_path

	with tqdm(desc="Downloading " + url, unit='iB', unit_scale=True) as pbar:
		response = requests.get(url, stream=True)

		total_length = int(response.headers.get('content-length', 0))
		pbar.reset(total=total_length)

		try:
			with open(download_path, "wb") as f:
				for data in response.iter_content(chunk_size=4096):
					pbar.update(len(data))
					f.write(data)
		except:
			download_path.unlink(missing_ok=True)
			raise

	if download_path != result_path:
		with zipfile.ZipFile(download_path, 'r') as zf:
			zf.extractall(result_path.parent)
		download_path.unlink()

	return result_path if result_path.exists() else None


def _normalize_word2vec(tokens, embeddings, normalizer, sampling='nearest'):
	if sampling not in ('nearest', 'average'):
		raise ValueError(f'Expected "nearest" or "average", got "{sampling}"')

	embeddings = embeddings.astype(np.float32)

	f_mask = np.zeros((embeddings.shape[0],), dtype=np.bool)
	f_tokens = []
	token_to_ids = dict()

	for i, t in enumerate(tqdm(tokens, desc="Normalizing Tokens")):
		nt = normalizer(t)
		if nt is None:
			continue
		if sampling != 'average' and nt != t:
			continue
		indices = token_to_ids.get(nt)
		if indices is None:
			token_to_ids[nt] = [i]
			f_tokens.append(nt)
			f_mask[i] = True
		else:
			indices.append(i)

	if sampling == 'average':
		for indices in tqdm(token_to_ids.values(), desc="Merging Tokens", total=len(token_to_ids)):
			if len(indices) > 1:
				i = indices[0]
				embeddings[i] = np.mean(embeddings[indices], axis=0)

	f_embeddings = embeddings[f_mask]
	embeddings = None

	assert f_embeddings.shape[0] == len(f_tokens)

	return f_tokens, f_embeddings


def _load_glove_txt(csv_path):
	tokens = []
	with open(csv_path, "r") as f:
		text = f.read()

	lines = text.split("\n")
	n_rows = len(lines)
	n_cols = len(lines[0].strip().split()) - 1

	embeddings = np.empty(
		shape=(n_rows, n_cols), dtype=np.float32)

	for line in tqdm(lines, desc="Importing " + str(csv_path)):
		values = line.strip().split()
		if values:
			t = values[0]
			if t:
				embeddings[len(tokens), :] = values[1:]
				tokens.append(t)

	embeddings = embeddings[:len(tokens), :]

	return tokens, embeddings


class Embedding:
	@property
	def is_static(self):
		return False

	@property
	def is_contextual(self):
		return False

	def create_instance(self):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class StaticEmbedding(Embedding):
	@property
	def is_static(self):
		return True

	def create_instance(self, session):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class AbstractVectors:
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
			raise RuntimeError("unsupported stacked source")

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


class StaticEmbeddingInstance:
	@property
	def is_static(self):
		return True


class CachedWordEmbedding(StaticEmbedding):
	class Cache:
		def __init__(self):
			import sqlite3

			self._cache_path = _make_cache_path() / 'cache'
			self._cache_path.mkdir(exist_ok=True, parents=True)

			self._conn = sqlite3.connect(self._cache_path / 'info.db')
			self._conn.execute("create table if not exists cache (key varchar primary key, path varchar)")

		def get(self, key):
			cur = self._conn.cursor()
			try:
				cur.execute('select path from cache where key=?', (key, ))
				r = cur.fetchone()
			finally:
				cur.close()

			if r is None:
				return None
			else:
				return self._cache_path / Path(r[0])

		def create_new_data_path(self):
			return self._cache_path / f"{time.time_ns()}.dat"

		def put(self, key, path):
			with self._conn:
				self._conn.execute("insert into cache(key, path) values (?, ?)", (
					key, str(path.relative_to(self._cache_path))))

	class Instance(StaticEmbeddingInstance):
		def __init__(self, name, tokens, vectors):
			self._name = name
			self._token2id = dict((t, i) for i, t in enumerate(tokens))
			self._vectors = vectors

		@property
		def memory_usage(self):
			return sys.getsizeof(self._token2id) + sys.getsizeof(self._vectors)

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			k = self._token2id.get(t)
			if k is not None:
				return self._vectors[k]
			else:
				return np.zeros((self.dimension,), dtype=np.float32)

		@property
		def dimension(self):
			return self._vectors.shape[1]

		def get_embeddings(self, tokens):
			indices = np.array([self._token2id.get(t, -1) for t in tokens], dtype=np.int32)
			oov = indices < 0
			data = self._vectors[np.maximum(indices, 0)].copy()
			data[oov, :].fill(0)  # now zero out those elements that are actually oov
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, embedding_sampling):
		self._loaded = {}
		self._embedding_sampling = embedding_sampling
		self._cache = CachedWordEmbedding.Cache()
		self._cache_path = _make_cache_path()

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):  # i.e. display name
		return self.unique_name

	@property
	def unique_name(self):
		raise NotImplementedError()

	def create_instance(self, session):
		normalizer = session.normalizer('text').to_callable()
		key = json.dumps({
			'emb': self.unique_name,
			'nrm': normalizer.ident,
			'sampling': self._embedding_sampling
		})

		loaded = self._loaded.get(key)
		if loaded is None:
			name = self.unique_name

			dat_path = self._cache.get(key)

			#dat_path = normalized_cache_path / f"{name}-{normalizer.name}-{self._embedding_sampling}.dat"

			if dat_path and dat_path.exists():
				with tqdm(desc="Opening " + self.name, total=1,  bar_format='{l_bar}{bar}') as pbar:
					with open(dat_path.with_suffix('.json'), 'r') as f:
						data = json.loads(f.read())
					tokens = data['tokens']
					vectors_mmap = np.memmap(
						dat_path, dtype=np.float32, mode='r', shape=tuple(data['shape']))
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				tokens, vectors = _normalize_word2vec(
					tokens, vectors, normalizer.unpack(), self._embedding_sampling)

				dat_path = self._cache.create_new_data_path()

				vectors_mmap = np.memmap(
					dat_path, dtype=np.float32, mode='w+', shape=vectors.shape)
				vectors_mmap[:, :] = vectors[:, :]
				vectors = None

				with open(dat_path.with_suffix('.json'), 'w') as f:
					f.write(json.dumps({
						'tokens': tokens,
						'shape': tuple(vectors_mmap.shape)
					}))

				self._cache.put(key, dat_path)

			loaded = CachedWordEmbedding.Instance(name, tokens, vectors_mmap)
			self._loaded[key] = loaded

		return loaded


class Word2VecVectors(CachedWordEmbedding):
	def __init__(self, name, path, binary=False, embedding_sampling="nearest"):
		super().__init__(embedding_sampling)
		self._name = name
		self._path = path
		self._binary = binary

	def _load(self):
		wv = gensim.models.KeyedVectors.load_word2vec_format(
			self._path, binary=self._binary)
		if _gensim_version < 4:
			emb_keys = wv.vocab
			emb_vectors = wv.vectors
		else:
			emb_keys = []
			emb_vectors = []
			vectors = wv.vectors
			for k, i in wv.key_to_index.items():
				emb_keys.append(k)
				emb_vectors.append(vectors[i])
		return emb_keys, emb_vectors

	@property
	def unique_name(self):
		return self._name


class KeyedVectors(StaticEmbedding):
	# using this class directly circumvents Vectorian's token
	# normalization. use with care.

	class Instance(StaticEmbeddingInstance):
		def __init__(self, name, wv):
			self._name = name
			self._wv = wv

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return self._wv.word_vec(t).astype(np.float32)

		@property
		def dimension(self):
			return self._wv.vector_size

		def get_embeddings(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in tqdm(enumerate(tokens), disable=len(tokens) < 1000):
				data[i, :] = self._wv.word_vec(t)
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, name, wv):
		self._name = name
		self._wv = wv

	def create_instance(self, session):
		return KeyedVectors.Instance(self._name, self._wv)

	@property
	def name(self):
		return self._name

	@staticmethod
	def load(path):
		wv = gensim.models.KeyedVectors.load(path)
		return KeyedVectors(Path(path).stem, wv)


class CompressedFastTextVectors(StaticEmbedding):
	def __init__(self, path):
		self._name = Path(path).stem
		self._wv = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(path))

	def create_instance(self, session):
		return KeyedVectors.Instance(self._name, self._wv)

	@property
	def name(self):
		return self._name


class Zoo:
	_initialized = False

	_embeddings = {
		'fasttext-en-mini': {
			'constructor': CompressedFastTextVectors,
			'url': 'https://www.dropbox.com/s/zj34mh3zp0pv1db/fasttext-en-mini.kv?dl=1'
		},
		'fasttext-de-mini': {
			'constructor': CompressedFastTextVectors,
			'url': 'https://www.dropbox.com/s/5u1742sqdww93ww/fasttext-de-mini.kv?dl=1'
		},
		'numberbatch-19.08-en-50': {
			'constructor': Word2VecVectors,
			'url': 'https://www.dropbox.com/s/9u29dsk1e0by9no/numberbatch-19.08-en-50.txt.zip?dl=1',
			'name': 'numberbatch-19.08-en-50'
		},
		'numberbatch-19.08-de-50': {
			'constructor': Word2VecVectors,
			'url': 'https://www.dropbox.com/s/yyha7lw72r11z3f/numberbatch-19.08-de-50.txt.zip?dl=1',
			'name': 'numberbatch-19.08-de-50'
		}
	}

	@staticmethod
	def _init():
		if Zoo._initialized:
			return

		from fasttext.util.util import valid_lang_ids as ft_valid_lang_ids
		for lang in ft_valid_lang_ids:
			Zoo._embeddings[f'fasttext-{lang}'] = {
				'constructor': PretrainedFastText,
				'lang': lang
			}

		for name, sizes in {
			'6B': [50, 100, 200, 300],
			'42B': [300],
			'840B': [300],
			'twitter.27B': [25, 50, 100, 200]}.items():

			for size in sizes:
				Zoo._embeddings[f'glove-{name}-{size}'] = {
					'constructor': PretrainedGloVe,
					'name': name,
					'ndims': size
				}

		Zoo._initialized = True

	@staticmethod
	def _download(url, force_download=False):
		cache_path = _make_cache_path()

		download_path = cache_path / "models"
		download_path.mkdir(exist_ok=True, parents=True)

		return _download(url, download_path, force_download=force_download)

	@staticmethod
	def list():
		Zoo._init()
		return tuple(sorted(Zoo._embeddings.keys()))

	@staticmethod
	def load(name, force_download=False):
		Zoo._init()
		spec = Zoo._embeddings.get(name)
		if spec is None:
			raise ValueError(f"unknown embedding name {name}")
		kwargs = dict((k, v) for k, v in spec.items() if k not in ("constructor", "url"))
		if "url" in spec:
			kwargs["path"] = Zoo._download(spec["url"], force_download)
		return spec["constructor"](**kwargs)


class ProgressParser(io.StringIO):
	def __init__(self, pbar):
		super().__init__()
		self._pbar = pbar
		self._pattern = re.compile(r"\((\d+\.\d+)%\)")

	def flush(self):
		s = self.getvalue()
		self.truncate(0)
		self.seek(0)

		m = self._pattern.search(s)
		if m:
			ratio = json.loads(m.group(1)) / 100
			self._pbar.n = int(self._pbar.total * ratio)
			self._pbar.refresh()


class PretrainedFastText(StaticEmbedding):
	class Instance(StaticEmbeddingInstance):
		def __init__(self, name, ft):
			self._name = name
			self._ft = ft

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return self._ft.get_word_vector(t)

		@property
		def dimension(self):
			return self._ft.get_dimension()

		def get_embeddings(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in tqdm(enumerate(tokens), disable=len(tokens) < 1000):
				data[i, :] = self._ft.get_word_vector(t)
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, lang):
		"""
		:param lang: language code of precomputed fasttext encodings, see
		https://fasttext.cc/docs/en/crawl-vectors.html
		"""

		import fasttext
		import fasttext.util

		super().__init__()
		self._lang = lang

		download_path = _make_cache_path() / 'models'
		download_path.mkdir(exist_ok=True, parents=True)

		filename = "cc.%s.300.bin" % self._lang
		if not (download_path / filename).exists():
			os.chdir(download_path)

			with tqdm(
					desc="Downloading " + self.name,
					total=10000,
					bar_format='{desc:<30}{percentage:3.2f}%|{bar:40}') as pbar:
				with contextlib.redirect_stdout(ProgressParser(pbar)):
					filename = fasttext.util.download_model(
						self._lang, if_exists='ignore')

		with tqdm(desc="Opening " + self.name, total=1, bar_format='{l_bar}{bar}') as pbar:
			ft = fasttext.load_model(str(download_path / filename))
			pbar.update(1)

		self._ft = ft

	def create_instance(self, session):
		return PretrainedFastText.Instance(self.name, self._ft)

	@property
	def name(self):
		return f"fasttext-{self._lang}"


class PretrainedGloVe(CachedWordEmbedding):
	def __init__(self, name="6B", ndims=300, embedding_sampling="nearest"):
		super().__init__(embedding_sampling)
		self._glove_name = name
		self._ndims = ndims

	def _load(self, force_download=False):
		download_path = self._cache_path / "models"
		download_path.mkdir(exist_ok=True, parents=True)
		txt_path = download_path / f"glove.{self._glove_name}.{self._ndims}d.txt"

		if force_download or not txt_path.exists():
			url = f"http://downloads.cs.stanford.edu/nlp/data/glove.{self._glove_name}.zip"
			_download(url, download_path, force_download=force_download)

		return _load_glove_txt(txt_path)

	@property
	def unique_name(self):
		return f"glove-{self._glove_name}-{self._ndims}"


class StackedEmbedding:
	class Instance(StaticEmbeddingInstance):
		def __init__(self, name, embeddings):
			self._name = name
			self._embeddings = embeddings

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return np.hstack([e.word_vec(t) for e in self.embeddings])

		@property
		def dimension(self):
			return sum(e.dimension for e in self._embeddings)

		def get_embeddings(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in tqdm(enumerate(tokens), disable=len(tokens) < 1000):
				data[i, :] = self.word_vec(t)
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, embeddings, name=None):
		if name is None:
			name = '[' + ', '.join([e.name for e in embeddings]) + ']'

		self._embeddings = embeddings
		self._name = name

	def create_instance(self, session):
		return StackedEmbedding.Instance(
			self.name, [e.create_instance(session) for e in self._embeddings])

	@property
	def name(self):
		return self._name


class Transform:
	def apply(self, vectors):
		raise NotImplementedError

	@property
	def name(self):
		raise NotImplementedError


class PCACompression(Transform):
	def __init__(self, n_dims):
		self._n_dims = n_dims

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


class ContextualEmbedding(Embedding):
	def __init__(self, transform=None):
		self._transform = transform

	@property
	def transform(self):
		return self._transform

	@property
	def is_contextual(self):
		return True

	def encode(self, doc):
		raise NotImplementedError()

	def create_instance(self, session):
		# ContextualEmbeddings are their own instance
		return self

	def to_core(self):
		return core.ContextualEmbedding(self.name)

	@property
	def name(self):
		raise NotImplementedError()


class SpacyTransformerEmbedding(ContextualEmbedding):
	def __init__(self, nlp, transform=None):
		super().__init__(transform)
		self._nlp = nlp

	def compressed(self, n_dims):
		return SpacyTransformerEmbedding(self._nlp, PCACompression(n_dims))

	def encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		# https://explosion.ai/blog/spacy-transformers
		# https://github.com/explosion/spaCy/issues/6403
		# https://github.com/explosion/spaCy/issues/7032
		# https://github.com/explosion/spaCy/discussions/6511

		if not hasattr(doc._, 'trf_data'):
			raise RuntimeError(
				"Could not access spaCy Transformer data for document. "
				f"Are you sure {self._nlp.meta['name']} is a Transformer model?")

		token_emb, sent_emb = doc._.trf_data.tensors
		token_emb = token_emb.reshape(-1, token_emb.shape[-1])
		n_dims = token_emb.shape[-1]

		trf_vectors = []

		assert len(doc) == len(doc._.trf_data.align)
		for x in doc._.trf_data.align:
			trf_vector = [token_emb[i[0]] for i in x.data]
			if trf_vector:
				trf_vectors.append(np.average(trf_vector, axis=0))
			else:
				trf_vectors.append(np.zeros((n_dims,), dtype=np.float32))

		trf_vectors = np.array(trf_vectors)
		assert len(doc) == trf_vectors.shape[0]

		return trf_vectors

	@cached_property
	def name(self):
		return '/'.join([
			'spacy', self._nlp.meta['name'], self._nlp.meta['version']
		] + ([] if self._transform is None else [self._transform.name]))


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
		raise vectors


class VectorsCache:
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


class ExternalMemoryVectorsRef(VectorsRef):
	def __init__(self, path):
		self._path = path

	def open(self):
		return ExternalMemoryVectors.load(self._path)


class MaskedVectorsRef(VectorsRef):
	def __init__(self, vectors, mask):
		self._vectors = vectors
		self._mask = mask

	def open(self):
		return MaskedVectors(
			self._vectors.open(), self._mask)

# === sentence/partition encoders.


def chunks(x, n):
	for i in range(0, len(x), n):
		yield x[i:i + n]


class PartitionEncoder:
	def __init__(self, partition, vector_size, cache_size=150, normalize=True):
		self._partition = partition
		self._cache = cachetools.LRUCache(cache_size)
		self._vector_size = vector_size

	@property
	def partition(self):
		return self._partition

	def prepare(self, docs, pbar=True):
		if len(docs) > self._cache.maxsize:
			raise RuntimeError("cache too small")
		self.encode(docs, pbar, update_cache=True)

	def _encode(self, docs, pbar):
		raise NotImplementedError()

	def encode(self, docs, pbar=False, update_cache=True):
		out = np.empty((len(docs), self._vector_size))

		new = []
		index = []

		for i, doc in enumerate(docs):
			cached = self._cache.get(doc.unique_id)
			if cached is not None:
				out[i, :] = cached
			else:
				new.append(doc)
				index.append(i)

		if new:
			for i, v in zip(index, self._encode(new, pbar)):
				out[i, :] = v
				if update_cache:
					uid = docs[i].unique_id
					if uid is not None:
						self._cache[uid] = v

		return Vectors(out)


class TokenAveragingEncoder(PartitionEncoder):
	# simple unweighted token averaging as described by Mikolov et al.
	# in "Distributed representations of words and phrases and their
	# compositionality.", 2013.

	def __init__(self, partition, embedding, cache_size=150):
		super().__init__(partition, embedding.dimension, cache_size)
		self._partition = partition
		self._embedding = embedding

	def _encode(self, docs, pbar):
		embeddings = []
		for doc in tqdm(docs, desc="Encoding", disable=not pbar):
			v = []
			for span in doc.spans(self._partition):
				for k, token in enumerate(span):
					v.append(self._embedding.word_vec(token.text))
			embeddings.append(np.mean(v, axis=0))
		return embeddings


class PartitionTextEncoder(PartitionEncoder):
	def __init__(self, partition, vector_size, chunk_size=50, cache_size=150):
		super().__init__(partition, vector_size, cache_size)
		self._chunk_size = chunk_size

	def _encode_text(self, text):
		raise NotImplementedError()

	def _encode(self, docs, pbar):
		n_spans = sum(doc.n_spans(self._partition) for doc in docs)

		embeddings = []
		with tqdm(desc="Encoding", total=n_spans, disable=not pbar) as pbar:
			for i, doc in enumerate(docs):
				spans = list(doc.spans(self._partition))
				for chunk in chunks(spans, self._chunk_size):
					doc_vec = self._encode_text([span.text for span in chunk])
					embeddings.append(doc_vec)
					pbar.update(len(chunk))

		return np.vstack(embeddings) if embeddings else []


class SentenceBERTEncoder(PartitionTextEncoder):
	# see https://github.com/UKPLab/sentence-transformers and
	# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
	# a good default is paraphrase-distilroberta-base-v1

	def __init__(self, partition, name, vector_size=768, **kwargs):
		super().__init__(partition, vector_size, **kwargs)

		from sentence_transformers import SentenceTransformer
		self._model = SentenceTransformer(name)

	def _encode_text(self, texts):
		return self._model.encode(texts)


# === utilities.

def extract_numberbatch(path, languages):
	# e.g. extract_numberbatch("/path/to/numberbatch-19.08.txt", ["en", "de"])
	# then use KeyedVectors.load()

	path = Path(path)
	languages = set(languages)

	pattern = re.compile(r"^/c/([a-z]+)/")

	with open(path, "r") as f:
		num_lines, num_dimensions = [int(x) for x in f.readline().split()]
		vectors = collections.defaultdict(lambda: {
			"keys": [],
			"vectors": []
		})

		for _ in tqdm(range(num_lines)):
			line = f.readline()
			m = pattern.match(line)
			if m:
				lang = m.group(1)
				if lang in languages:
					line = line[len(m.group(0)):]
					cols = line.split()
					key = cols[0]
					if key.isalpha():
						record = vectors[lang]
						record["keys"].append(key)
						record["vectors"].append(
							np.array([float(x) for x in cols[1:]]))

	for lang, record in vectors.items():
		wv = gensim.models.KeyedVectors(num_dimensions)
		wv.add_vectors(record["keys"], record["vectors"])
		wv.save(str(path.parent / f"{path.stem}-{lang}.kv"))


def compress_keyed_vectors(model, n_dims):
	pca = sklearn.decomposition.PCA(n_components=n_dims)
	vectors = pca.fit_transform(model.vectors)
	wv = gensim.models.KeyedVectors(
		vector_size=n_dims, count=vectors.shape[0])
	if _gensim_version < 4:
		for i, k in enumerate(model.vocab):
			wv.add_vector(k, vectors[i])
	else:
		for k, i in model.key_to_index.items():
			wv.add_vector(k, vectors[i])
	return wv
