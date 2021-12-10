import sqlite3
import sys
import numpy as np
import gensim
import gensim.models
import gensim.downloader
import time
import json
import vectorian.core as core
from pathlib import Path
from vectorian.tqdm import tqdm
from .token import TokenEmbedding
from ..vectors import Vectors
from ..utils import make_cache_path, normalize_word2vec, extraction_tqdm, gensim_version
from ..transform import PCACompression
from ..encoder import EmbeddingEncoder


class StaticEmbedding(TokenEmbedding):
	@property
	def is_static(self):
		return True

	def create_encoder(self, session):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()


class StaticEmbeddingEncoder(EmbeddingEncoder):
	@property
	def is_static(self):
		return True

	def encode_tokens(self, tokens):
		raise NotImplementedError()


class CachedWordEmbedding(StaticEmbedding):
	pbar_on_open = False

	class Cache:
		def __init__(self):
			self._cache_path = make_cache_path() / 'cache'
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

	class Encoder(StaticEmbeddingEncoder):
		def __init__(self, embedding, name, tokens, vectors):
			self._embedding = embedding
			self._name = name
			self._token2id = dict((t, i) for i, t in enumerate(tokens))
			self._vectors = vectors

		@property
		def embedding(self):
			return self._embedding

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

		def encode_tokens(self, tokens):
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
		self._cache_path = make_cache_path()
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

	def create_encoder(self, session):
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
				with tqdm(desc="Opening " + self.name, total=1,  bar_format='{l_bar}{bar}',
						  disable=not CachedWordEmbedding.pbar_on_open) as pbar:
					with open(dat_path.with_suffix('.json'), 'r') as f:
						data = json.loads(f.read())
					tokens = data['tokens']
					vectors_mmap = np.memmap(
						dat_path, dtype=np.float32, mode='r', shape=tuple(data['shape']))
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				tokens, vectors = normalize_word2vec(
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

			loaded = CachedWordEmbedding.Encoder(
				self, name, tokens, vectors_mmap)
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
		if gensim_version() < 4:
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


class OneHotEncoding(StaticEmbedding):
	class Encoder(StaticEmbeddingEncoder):
		pass

	def create_encoder(self, session):
		return OneHotEncoding.Encoder()

	@property
	def name(self):
		return "one-hot"


class KeyedVectors(StaticEmbedding):
	# using this class directly circumvents Vectorian's token
	# normalization. use with care.

	class Encoder(StaticEmbeddingEncoder):
		def __init__(self, embedding, name, wv):
			self._embedding = embedding
			self._name = name
			self._wv = wv

		@property
		def embedding(self):
			return self._embedding

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return self._wv.word_vec(t).astype(np.float32)

		@property
		def dimension(self):
			return self._wv.vector_size

		def encode_tokens(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in enumerate(extraction_tqdm(tokens, self.name)):
				data[i, :] = self._wv.word_vec(t)
			return Vectors(data)

		def to_core(self, tokens):
			return core.StaticEmbedding(self, tokens)

	def __init__(self, name, wv):
		self._name = name
		self._wv = wv

	def create_encoder(self, session):
		return KeyedVectors.Encoder(self, self._name, self._wv)

	@property
	def name(self):
		return self._name

	@staticmethod
	def load(path):
		wv = gensim.models.KeyedVectors.load(path)
		return KeyedVectors(Path(path).stem, wv)


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
			download(url, download_path, force_download=force_download)

		return load_glove_txt(txt_path)

	@property
	def unique_name(self):
		return f"glove-{self._glove_name}-{self._ndims}"


class StackedEmbedding(StaticEmbedding):
	class Encoder(StaticEmbeddingEncoder):
		def __init__(self, embedding, name, embeddings):
			self._embedding = embedding
			self._name = name
			self._embeddings = embeddings

		@property
		def embedding(self):
			return self._embedding

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return np.hstack([e.word_vec(t) for e in self._embeddings])

		@property
		def dimension(self):
			return sum(e.dimension for e in self._embeddings)

		def encode_tokens(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in enumerate(extraction_tqdm(tokens, self.name)):
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

	def create_encoder(self, session):
		return StackedEmbedding.Encoder(
			self, self.name, [e.create_encoder(session) for e in self._embeddings])

	@property
	def name(self):
		return self._name
