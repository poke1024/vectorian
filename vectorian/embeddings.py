import vectorian.core as core

from tqdm import tqdm
from pathlib import Path
from cached_property import cached_property

import numpy as np
import json
import os
import download
import compress_fasttext


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


def _make_cache_path():
	cache_path = Path.home() / ".vectorian" / "embeddings"
	cache_path.mkdir(exist_ok=True, parents=True)
	return cache_path


class StaticEmbedding:
	def create_instance(self, session):
		raise NotImplementedError()


class Vectors:  # future: CudaVectors
	def __init__(self, unmodified):
		self._unmodified = unmodified

	@property
	def size(self):
		return self._unmodified.shape[0]

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


class StackedVectors:
	def __init__(self, sources, indices):
		self._sources = sources
		self._indices = indices

	@cached_property
	def size(self):
		return self.unmodified.shape[0]

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


class CachedWordEmbedding(StaticEmbedding):
	class Instance:
		def __init__(self, name, tokens, vectors):
			self._name = name
			self._token2id = dict((t, i) for i, t in enumerate(tokens))
			self._vectors = vectors

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
		self._cache_path = _make_cache_path()
		self._embedding_sampling = embedding_sampling

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):  # i.e. display name
		return self.unique_name

	@property
	def unique_name(self):
		raise NotImplementedError()

	def create_instance(self, session):
		normalizer = session.token_mapper('tokenizer')

		loaded = self._loaded.get(normalizer.name)
		if loaded is None:
			name = self.unique_name

			normalized_cache_path = self._cache_path / 'cache'
			normalized_cache_path.mkdir(exist_ok=True, parents=True)
			dat_path = normalized_cache_path / f"{name}-{normalizer.name}-{self._embedding_sampling}.dat"

			if dat_path.exists():
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

				vectors_mmap = np.memmap(
					dat_path, dtype=np.float32, mode='w+', shape=vectors.shape)
				vectors_mmap[:, :] = vectors[:, :]
				vectors = None

				with open(dat_path.with_suffix('.json'), 'w') as f:
					f.write(json.dumps({
						'tokens': tokens,
						'shape': tuple(vectors_mmap.shape)
					}))

			loaded = CachedWordEmbedding.Instance(name, tokens, vectors_mmap)
			self._loaded[normalizer.name] = loaded

		return loaded


class GensimKeyedVectors(StaticEmbedding):
	class Instance:
		def __init__(self, name, wv):
			self._name = name
			self._wv = wv

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return self._wv.word_vec(t)

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
		return GensimKeyedVectors.Instance(self._name, self._wv)

	@property
	def name(self):
		return self._name

	@staticmethod
	def load(path):
		from gensim.models import KeyedVectors
		wv = KeyedVectors.load(path)
		return GensimKeyedVectors(Path(path).stem, wv)


class CompressedFastTextVectors(StaticEmbedding):
	def __init__(self, path):
		self._name = Path(path).stem
		self._wv = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path)

	def create_instance(self, session):
		return GensimKeyedVectors.Instance(self._name, self._wv)

	@property
	def name(self):
		return self._name


class PretrainedFastText(StaticEmbedding):
	class Instance:
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
			with tqdm(desc="Downloading " + self.name, total=1, bar_format='{l_bar}{bar}') as pbar:
				filename = fasttext.util.download_model(
					self._lang, if_exists='ignore')
				pbar.update(1)

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

	def _load(self):
		download_path = self._cache_path / "models"
		download_path.mkdir(exist_ok=True, parents=True)

		txt_data_path = download_path / f"glove-{self._glove_name}"

		if not txt_data_path.exists():
			url = f"http://downloads.cs.stanford.edu/nlp/data/glove.{self._glove_name}.zip"
			download.download(url, txt_data_path, kind="zip", progressbar=True)

		return _load_glove_txt(
			txt_data_path / f"glove.{self._glove_name}.{self._ndims}d.txt")

	@property
	def unique_name(self):
		return f"glove-{self._glove_name}-{self._ndims}"


class StackedEmbedding:
	class Instance:
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


class ContextualEmbedding:
	def encode(self, doc):
		raise NotImplementedError()


class TransformerEmbedding(ContextualEmbedding):
	def encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		return doc._.trf_data.tensors[-1]
