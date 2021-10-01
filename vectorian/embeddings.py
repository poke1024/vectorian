import vectorian.core as core

from tqdm.autonotebook import tqdm
from pathlib import Path
from cached_property import cached_property
from functools import partial

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
import gensim.downloader
import requests
import zipfile
import urllib.parse
import cachetools
import sqlite3
import uuid


pbar_on_open = False


_gensim_version = int(gensim.__version__.split(".")[0])
if _gensim_version >= 4:
	raise RuntimeError("Vectorian needs gensim < 4.0.0")


_custom_cache_path = os.environ.get("VECTORIAN_CACHE_HOME")


def set_cache_path(path):
	global _custom_cache_path
	_custom_cache_path = Path(path)


def _make_cache_path():
	if _custom_cache_path is None:
		cache_path = Path.home() / ".vectorian" / "embeddings"
	else:
		cache_path = Path(_custom_cache_path) / "embeddings"
	cache_path.mkdir(exist_ok=True, parents=True)
	return cache_path


def _extraction_tqdm(tokens, name):
	return tqdm(tokens, desc=f"Extracting {name}", disable=len(tokens) < 5000)


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
			for zi in zf.infolist():
				if zi.filename[-1] == '/':
					continue
				zi.filename = os.path.basename(zi.filename)
				p = zf.extract(zi, result_path.parent)
				Path(p).rename(result_path)

		download_path.unlink()

	return result_path if result_path.exists() else None


def _normalize_word2vec(name, tokens, embeddings, normalizer, sampling='nearest'):
	if sampling not in ('nearest', 'average'):
		raise ValueError(f'Expected "nearest" or "average", got "{sampling}"')

	embeddings = embeddings.astype(np.float32)

	f_mask = np.zeros((embeddings.shape[0],), dtype=np.bool)
	f_tokens = []
	token_to_ids = dict()

	for i, t in enumerate(tqdm(tokens, desc=f"Normalizing tokens in {name}")):
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
		for indices in tqdm(token_to_ids.values(), desc=f"Merging tokens in {name}", total=len(token_to_ids)):
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

	def __init__(self, embedding_sampling, transforms=None):
		self._loaded = {}
		self._embedding_sampling = embedding_sampling
		self._cache = CachedWordEmbedding.Cache()
		self._cache_path = _make_cache_path()
		self._transforms = transforms or []

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):  # i.e. display name
		return self.unique_name

	@property
	def unique_name(self):
		raise NotImplementedError()

	@property
	def constructor_args(self):
		return {
			'embedding_sampling': self._embedding_sampling,
			'transforms': self._transforms
		}

	def pca(self, n_dims):
		kwargs = self.constructor_args
		kwargs['transforms'] = kwargs['transforms'] + [PCACompression(n_dims)]
		return self.__class__(**kwargs)

	def create_instance(self, session):
		normalizers = session.normalizers
		normalizer = normalizers['text'].to_callable()
		key = json.dumps({
			'emb': self.unique_name,
			'nrm': normalizer.ident,
			'sampling': self._embedding_sampling,
			'tfm': [t.name for t in self._transforms]
		})

		loaded = self._loaded.get(key)
		if loaded is None:
			name = self.unique_name

			dat_path = self._cache.get(key)

			#dat_path = normalized_cache_path / f"{name}-{normalizer.name}-{self._embedding_sampling}.dat"

			if dat_path and dat_path.exists():
				with tqdm(desc="Opening " + self.name, total=1,  bar_format='{l_bar}{bar}', disable=not pbar_on_open) as pbar:
					with open(dat_path.with_suffix('.json'), 'r') as f:
						data = json.loads(f.read())
					tokens = data['tokens']
					vectors_mmap = np.memmap(
						dat_path, dtype=np.float32, mode='r', shape=tuple(data['shape']))
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				tokens, vectors = _normalize_word2vec(
					self.name, tokens, vectors, normalizer.unpack(), self._embedding_sampling)

				for t in self._transforms:
					vectors = t.apply(Vectors(vectors)).unmodified

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


class GensimVectors(CachedWordEmbedding):
	def __init__(self, name, embedding_sampling="nearest", **kwargs):
		super().__init__(embedding_sampling, **kwargs)
		self._name = name

	@property
	def constructor_args(self):
		return dict({
			'name': self._name
		}, **super().constructor_args)

	def _load_wv(self):
		raise NotImplementedError()

	def _load(self):
		wv = self._load_wv()
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


class PretrainedGensimVectors(GensimVectors):
	def __init__(self, name, gensim_name, embedding_sampling="nearest", **kwargs):
		super().__init__(name, embedding_sampling, **kwargs)
		self._gensim_name = gensim_name

	@property
	def constructor_args(self):
		return dict({
			'gensim_name': self._gensim_name
		}, **super().constructor_args)

	def _load_wv(self):
		return gensim.downloader.load(self._gensim_name)


class Word2VecVectors(GensimVectors):
	def __init__(self, name, path, binary=False, embedding_sampling="nearest", **kwargs):
		super().__init__(name, embedding_sampling, **kwargs)
		self._path = path
		self._binary = binary

	@property
	def constructor_args(self):
		return dict({
			'path': self._path,
			'binary': self._binary
		}, **super().constructor_args)

	def _load_wv(self):
		return gensim.models.KeyedVectors.load_word2vec_format(
			self._path, binary=self._binary)


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
			for i, t in enumerate(_extraction_tqdm(tokens, self.name)):
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


def _zenodo_url(record, name):
	return f'https://zenodo.org/record/{record}/files/{name}'


class Zoo:
	_initialized = False

	_numberbatch_lang_codes = [
		'af', 'ang', 'ar', 'ast', 'az', 'be', 'bg', 'ca', 'cs', 'cy', 'da',
		'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fo',
		'fr', 'fro', 'ga', 'gd', 'gl', 'grc', 'gv', 'he', 'hi', 'hsb', 'hu',
		'hy', 'io', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'ku', 'la', 'lt', 'lv',
		'mg', 'mk', 'ms', 'mul', 'nl', 'no', 'non', 'nrf', 'nv', 'oc', 'pl',
		'pt', 'ro', 'ru', 'rup', 'sa', 'se', 'sh', 'sk', 'sl', 'sq', 'sv', 'sw',
		'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'vo', 'xcl', 'zh'
	]

	_embeddings = {
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

			Zoo._embeddings[f'fasttext-{lang}-mini'] = {
				'constructor': CompressedFastTextVectors,
				'url': _zenodo_url(4905385, f'fasttext-{lang}-mini')
			}

		for lang in Zoo._numberbatch_lang_codes:
			Zoo._embeddings[f'numberbatch-19.08-{lang}'] = {
				'constructor': partial(Word2VecVectors, binary=True),
				'url': _zenodo_url(4911598, f'numberbatch-19.08-{lang}.zip'),
				'name': f'numberbatch-19.08-{lang}'
			}

		for d in [50, 100, 200, 300]:
			Zoo._embeddings[f'glove-6B-{d}'] = {
				'constructor': partial(Word2VecVectors, binary=True),
				'url': _zenodo_url(4925376, f'glove.6B.{d}d.zip'),
				'name': f'glove-6B-{d}'
			}

		for name, sizes in {
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
			for i, t in enumerate(_extraction_tqdm(tokens, self.name)):
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


class StackedEmbedding(StaticEmbedding):
	class Instance(StaticEmbeddingInstance):
		def __init__(self, name, embeddings):
			self._name = name
			self._embeddings = embeddings

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return np.hstack([e.word_vec(t) for e in self._embeddings])

		@property
		def dimension(self):
			return sum(e.dimension for e in self._embeddings)

		def get_embeddings(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in enumerate(_extraction_tqdm(tokens, self.name)):
				data[i, :] = self.word_vec(t)
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, embeddings, name=None):
		if name is None:
			name = ' + '.join([e.name for e in embeddings])

		if not all(e.is_static for e in embeddings):
			raise RuntimeError("currently StackedEmbedding only supports static embeddings")

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


class SpacyEmbedding(ContextualEmbedding):
	def __init__(self, nlp, transform=None):
		super().__init__(transform)
		self._nlp = nlp

	@property
	def nlp(self):
		return self._nlp

	@cached_property
	def name(self):
		meta = self._nlp.meta
		return '/'.join([
			meta['url'], meta['lang'], meta['name'], meta['version']
		] + ([] if self._transform is None else [self._transform.name]))


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


class SpacyVectorEmbedding(SpacyEmbedding):
	def __init__(self, nlp, dimension, cache=None, **kwargs):
		super().__init__(nlp, **kwargs)
		self._dimension = dimension
		self._cache = cache

	@property
	def dimension(self):
		return self._dimension

	def pca(self, n_dims):
		return SpacyVectorEmbedding(self._nlp, PCACompression(n_dims))

	def encode(self, doc):
		if self._cache is not None:
			array = self._cache.get(doc.text)
			if array is not None:
				return array

		array = np.array([token.vector for token in self._nlp(doc.text)])

		if self._cache is not None:
			self._cache.put(doc.text, array)

		return array


class SpacyTransformerEmbedding(SpacyEmbedding):
	@property
	def dimension(self):
		if self._transform is not None:
			return self._transform.dimension
		else:
			# https://spacy.io/usage/processing-pipelines
			# https://thinc.ai/docs/api-model
			tfm = self._nlp.pipeline[self._nlp.pipe_names.index("transformer")][1]
			return tfm.model.get_dim("nO")

	def pca(self, n_dims):
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


# === sentence/partition encoders.


def chunks(x, n):
	for i in range(0, len(x), n):
		yield x[i:i + n]


def _prepare_doc(doc, nlp):
	if hasattr(doc, 'prepare'):
		if nlp is None:
			raise RuntimeError(f"need nlp to prepare {doc}")
		return doc.prepare(nlp)
	else:
		return doc


def prepare_docs(docs, nlp):
	return [_prepare_doc(doc, nlp) for doc in docs]


class AbstractPartitionEncoder:
	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def embedding(self):
		raise NotImplementedError()

	def encode(self, docs, partition, pbar=False):
		raise NotImplementedError()


class PartitionEncoder(AbstractPartitionEncoder):
	def __init__(self, span_encoder):
		self._span_encoder = span_encoder

	def vector_size(self, session):
		return self._span_encoder.vector_size(session)

	@property
	def embedding(self):  # i.e. token embedding
		return self._span_encoder.embedding

	def encode(self, docs, partition, pbar=False):
		n_spans = [doc.n_spans(partition) for doc in docs]
		i_spans = np.cumsum([0] + n_spans)

		out = np.empty((i_spans[-1], self.vector_size(partition.session)))

		def gen_spans():
			with tqdm(
				desc="Encoding",
				total=i_spans[-1],
				disable=not pbar) as pbar_instance:

				for doc in docs:
					spans = list(doc.spans(partition))
					yield doc, spans
					pbar_instance.update(len(spans))

		for i, v in enumerate(self._span_encoder.encode(partition.session, gen_spans())):
			out[i_spans[i]:i_spans[i + 1], :] = v

		return Vectors(out)

	def to_cached(self, cache_size=150):
		return CachedPartitionEncoder(self._encoder, cache_size)


class CachedPartitionEncoder(AbstractPartitionEncoder):
	def __init__(self, span_encoder, cache_size=150):
		self._encoder = PartitionEncoder(span_encoder)
		self._cache = cachetools.LRUCache(cache_size)
		self._corpus = None

	def save(self, path):
		path = Path(path)
		index = []
		with h5py.File(path.parent / (path.name + ".h5"), 'w') as f:
			for k, v in self._cache.items():
				f.create_dataset(str(len(index)), data=v)
				index.append(k)
		with open(path.parent / (path.name + ".json"), "w") as f:
			f.write(json.dumps(index))

	def try_load(self, path):
		path = Path(path)
		if not (path.parent / (path.name + ".json")).exists():
			return False
		with open(path.parent / (path.name + ".json"), "r") as f:
			index = json.loads(f.read())
		if len(index) > self._cache.maxsize:
			raise RuntimeError("cache is too small")
		with h5py.File(path.parent / (path.name + ".h5"), 'r') as f:
			for i, key in enumerate(index):
				self._cache[tuple(key)] = np.array(f[str(i)])

		return True

	def load(self, path):
		if not self.try_load(path):
			raise FileNotFoundError(path)

	def cache(self, docs, partition, pbar=True):
		if len(docs) > self._cache.maxsize:
			raise RuntimeError("cache too small")
		self.encode(docs, partition, pbar=pbar)

	def vector_size(self, session):
		return self._encoder.vector_size(session)

	@property
	def embedding(self):
		return self._encoder.embedding

	def encode(self, docs, partition, pbar=False):
		n_spans = [doc.n_spans(partition) for doc in docs]
		i_spans = np.cumsum([0] + n_spans)

		out = np.empty((sum(n_spans), self.vector_size(partition.session)))

		new = []
		index = []

		if self._corpus is None:
			for doc in docs:
				if doc.corpus is not None:
					self._corpus = doc.corpus
					break

		# we assume all docs stem from the same corpus. otherwise our caching
		# ids would not be reliable.
		for doc in docs:
			if doc.corpus not in (None, self._corpus):
				raise RuntimeError(f"doc {doc} has corpus {doc.corpus}, expected either None or {self._corpus}")


		def mk_cache_key(doc):
			if doc.corpus is None:
				return None
			uid = doc.corpus_id
			if uid is None:
				return None
			return (uid,) + partition.cache_key

		for i, doc in enumerate(docs):
			cached = self._cache.get(mk_cache_key(doc))
			if cached is not None:
				out[i_spans[i]:i_spans[i + 1], :] = cached
			else:
				new.append(doc)
				index.append(i)

		if new:
			v = self._encoder.encode(new, partition, pbar).unmodified

			n_spans_new = [n_spans[i] for i in index]
			i_spans_new = np.cumsum([0] + n_spans_new)

			for j, i in enumerate(index):
				v_doc = v[i_spans_new[j]:i_spans_new[j + 1], :]
				out[i_spans[i]:i_spans[i + 1], :] = v_doc

				cache_key = mk_cache_key(new[i])
				if cache_key is not None:
					self._cache[cache_key] = v_doc

		return Vectors(out)

	def to_cached(self, cache_size=None):
		return self


class AbstractSpanEncoder:
	def __init__(self):
		pass

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def embedding(self):  # i.e. token embedding
		raise NotImplementedError()

	def encode(self, session, doc_spans):
		raise NotImplementedError()


class TokenEmbeddingAggregator(AbstractSpanEncoder):
	# simple aggregated token embeddings, e.g. unweighted token
	# averaging as described by Mikolov et al.
	# in "Distributed representations of words and phrases and their
	# compositionality.", 2013.

	def __init__(self, embedding, agg=np.mean):
		super().__init__()
		self._embedding = embedding
		self._agg = agg

		if embedding.is_contextual and embedding.transform is not None:
			raise RuntimeError("cannot use transformed contextual embedding with TokenAggregator")

	@property
	def embedding(self):
		return self._embedding

	def vector_size(self, session):
		return session.to_embedding_instance(self._embedding).dimension

	def encode(self, session, doc_spans):
		embedding = session.to_embedding_instance(self._embedding)

		for doc, spans in doc_spans:
			out = np.empty((len(spans), self.vector_size(session)), dtype=np.float32)
			out.fill(np.nan)

			if embedding.is_static:
				for i, span in enumerate(spans):
					text = [token.text for token in span]
					emb_vec = embedding.get_embeddings(text)
					v = emb_vec.unmodified
					if v.shape[0] > 0:
						out[i, :] = self._agg(v, axis=0)

			elif embedding.is_contextual:
				vec_ref = doc.contextual_embeddings[self._embedding.name]
				with vec_ref.open() as emb_vec:
					emb_vec_data = emb_vec.unmodified
					for i, span in enumerate(spans):
						v = emb_vec_data[span.start:span.end, :]
						if v.shape[0] > 0:
							out[i, :] = self._agg(v, axis=0)

			else:
				assert False

			yield out


class SpanTextEncoder(AbstractSpanEncoder):
	def __init__(self, chunk_size=50):
		super().__init__()
		self._chunk_size = chunk_size

	def _encode_text(self, text):
		raise NotImplementedError()

	def vector_size(self, session):
		raise NotImplementedError()

	def encode(self, session, doc_spans):
		for doc, spans in doc_spans:
			#for chunk in chunks(spans, self._chunk_size):
			yield self._encode_text([span.text for span in spans])


class SpanEncoder(SpanTextEncoder):
	def __init__(self, encode, vector_size=768, **kwargs):
		super().__init__(**kwargs)
		self._encode = encode
		self._vector_size = vector_size

	def vector_size(self, session):
		return self._vector_size

	@property
	def embedding(self):  # i.e. token embedding
		return None

	def _encode_text(self, texts):
		return self._encode(texts)


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
