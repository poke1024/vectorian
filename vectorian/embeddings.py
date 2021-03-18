import vectorian.core as core

from tqdm import tqdm
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import download
import logging
import compress_fasttext


def _make_table(tokens, embeddings, normalizer, sampling='nearest'):
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

	vecs = [pa.array(f_embeddings[:, i]) for i in range(f_embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(f_embeddings.shape[1])]
	f_embeddings = None

	return pa.Table.from_arrays(
		[pa.array(f_tokens, type=pa.string())] + vecs,
		['token'] + vecs_name)


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
	def create_instance(self, normalizer, embedding_sampling):
		raise NotImplementedError()


class WordEmbeddingCache:
	def __init__(self):
		self._loaded = {}
		self._cache_path = _make_cache_path()

	@property
	def get_dimension(self):
		return self._ndims

	def get_embeddings(self, tokens, array):
		for i, t in tqdm(enumerate(tokens)):
			array[i, :] = self._ft.get_word_vector(t)

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):  # i.e. display name
		return self.unique_name

	@property
	def unique_name(self):
		raise NotImplementedError()

	def create_instance(self, normalizer, embedding_sampling):
		loaded = self._loaded.get(normalizer.name)
		if loaded is None:
			name = self.unique_name

			normalized_cache_path = self._cache_path / 'parquet'
			normalized_cache_path.mkdir(exist_ok=True, parents=True)
			pq_path = normalized_cache_path / f"{name}-{normalizer.name}-{embedding_sampling}.parquet"

			if pq_path.exists():
				with tqdm(desc="Opening " + self.name, total=1,  bar_format='{l_bar}{bar}') as pbar:
					table = pq.read_table(pq_path, memory_map=True)
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				table = _make_table(
					tokens, vectors, normalizer.unpack(), embedding_sampling)

			loaded = StaticEmbeddingInstance(name, table)
			self._loaded[normalizer.name] = loaded

			if pq_path and not pq_path.exists():
				loaded.save(pq_path)

			loaded.free_table()

		return loaded


class EmbeddingLoadingProgress:
	def __init__(self, pbar):
		self._pbar = pbar
		self._last = 0

	def update(self, ratio):
		d = ratio - self._last
		self._last = ratio
		self._pbar.update(d)


class StaticEmbeddingInstance:
	pass


class StaticWordEmbeddingInstance(StaticEmbeddingInstance):
	def __init__(self, name, table):
		self._name = name
		self._table = table

		with tqdm(desc="Loading " + name, total=1, bar_format='{l_bar}{bar}') as pbar:
			progress = EmbeddingLoadingProgress(pbar)
			self._core = core.StaticEmbedding(
				self._name, table, progress.update)

		self._vec = self._core.vectors

	@property
	def name(self):
		return self._name

	def free_table(self):
		self._table = None

	def save(self, path):
		logging.info(f"writing {path}")
		pq.write_table(
			self._table,
			path,
			compression='none',
			version='2.0')
		logging.info("done.")

	def tok2vec(self, token, normalized=True):
		i = self._core.token_to_id(token)
		if i < 0:
			return None
		else:
			cat = "normalized" if normalized else "unmodified"
			return self._vec[cat][i]

	def to_core(self):
		return self._core


class InstalledStaticEmbedding(StaticEmbedding):
	def __init__(self, path, unique_name=None):
		super().__init__()

		if unique_name is None:
			unique_name = Path(path).stem

		self._path = path
		self._unique_name = unique_name

	@property
	def unique_name(self):
		return self._unique_name


class GensimKeyedVectors(InstalledStaticEmbedding):
	def _load(self):
		from gensim.models import KeyedVectors
		wv = KeyedVectors.load(self._path)
		return wv.index2word, wv.vectors_vocab


class FastTextVectors(InstalledStaticEmbedding):
	def _load(self):
		from gensim.models.fasttext import load_facebook_vectors
		wv = load_facebook_vectors(self._path)
		# see https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastTextKeyedVectors
		return wv.index2word, wv.vectors_vocab

	def save_compressed(self, path):
		from gensim.models.fasttext import load_facebook_vectors
		import compress_fasttext
		big_model = load_facebook_vectors(self._path)
		small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
		small_model.save(path)


class CompressedFastTextVectors(InstalledStaticEmbedding):
	def _load(self):
		wv = compress_fasttext.models.CompressedFastTextKeyedVectors.load(self._path)
		return wv.index2word, np.array([wv.word_vec(word) for word in wv.index2word])


class PretrainedFastText(StaticEmbedding):
	class Instance:
		def __init__(self, name, ft):
			self._name = name
			self._ft = ft

		@property
		def name(self):
			return self._name

		@property
		def dimension(self):
			return self._ft.get_dimension()

		def get_embeddings(self, tokens):
			print("get_embeddings", len(tokens))
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in tqdm(enumerate(tokens)):
				data[i, :] = self._ft.get_word_vector(t)
			return data

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


class GloVeCache(WordEmbeddingCache):
	def __init__(self, name="6B", ndims=300):
		super().__init__()
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


class PretrainedGloVe(StaticEmbedding):
	def __init__(self, name="6B", ndims=300):
		"""
		:param name: one of "6B", "42B.300d", "840B.300d",
		"twitter.27B", see https://nlp.stanford.edu/projects/glove/
		"""

		super().__init__()
		self._glove_name = name
		self._ndims = ndims

	def load(self, session):
		# normalizer, embedding_sampling,
		pass


	@property
	def unique_name(self):
		return f"glove-{self._glove_name}-{self._ndims}"


class ContextualEmbedding:
	def encode(self, doc):
		raise NotImplementedError()


class TransformerEmbedding(ContextualEmbedding):
	def encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		return doc._.trf_data.tensors[-1]
