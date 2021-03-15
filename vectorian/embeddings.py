import vectorian.core as core

from tqdm import tqdm
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import download
import logging


def _make_table(tokens, embeddings, normalizer):
	embeddings = embeddings.astype(np.float32)

	f_mask = np.zeros((embeddings.shape[0],), dtype=np.bool)
	f_tokens = []
	token_to_ids = dict()

	for i, t in enumerate(tqdm(tokens, desc="Normalizing Tokens")):
		nt = normalizer(t)
		if nt:
			indices = token_to_ids.get(nt)
			if indices is None:
				token_to_ids[nt] = [i]
				f_tokens.append(nt)
				f_mask[i] = True
			else:
				indices.append(i)

	for indices in tqdm(token_to_ids.values(), desc="Merging Tokens", total=len(token_to_ids)):
		if len(indices) > 1:
			i = indices[0]
			embeddings[i] = np.mean(embeddings[indices], axis=0)

	embeddings = embeddings[f_mask]
	assert embeddings.shape[0] == len(f_tokens)

	vecs = [pa.array(embeddings[:, i]) for i in range(embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(embeddings.shape[1])]

	return pa.Table.from_arrays(
		[pa.array(f_tokens, type=pa.string())] + vecs,
		['token'] + vecs_name)


def _load_fasttext_txt(csv_path):
	tokens = []
	with open(csv_path, "r") as f:
		n_rows, n_cols = map(int, f.readline().strip().split())

		embeddings = np.empty(
			shape=(n_rows, n_cols), dtype=np.float32)

		for _ in tqdm(range(n_rows), desc="Importing " + csv_path):
			values = f.readline().strip().split()
			if values:
				t = values[0]
				if t:
					embeddings[len(tokens), :] = values[1:]
					tokens.append(t)

	embeddings = embeddings[:len(tokens), :]

	return tokens, embeddings


def _load_glove_txt(csv_path):
	tokens = []
	with open(csv_path, "r") as f:
		text = f.read()

	lines = text.split("\n")
	n_rows = len(lines)
	n_cols = len(lines[0].strip().split()) - 1

	embeddings = np.empty(
		shape=(n_rows, n_cols), dtype=np.float32)

	for line in tqdm(lines, desc="Importing " + csv_path):
		values = line.strip().split()
		if values:
			t = values[0]
			if t:
				embeddings[len(tokens), :] = values[1:]
				tokens.append(t)

	embeddings = embeddings[:len(tokens), :]

	return tokens, embeddings


def cache_path(path, name):
	if name:
		return path.parent / (path.stem + "." + name + path.suffix)
	else:
		return path


class StaticEmbedding:
	def __init__(self, path, name):
		self._path = path
		self._name = name

		self._loaded = {}

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):
		return self._name

	def create_instance(self, normalizer):
		loaded = self._loaded.get(normalizer.name)
		if loaded is None:
			if self._path:
				path = cache_path(self._path, normalizer.name)
			else:
				path = None
			if path and path.exists():
				with tqdm(desc="Opening " + self._name, total=1,  bar_format='{l_bar}{bar}') as pbar:
					table = pq.read_table(path, memory_map=True)
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				table = _make_table(
					tokens, vectors, normalizer.unpack())

			loaded = StaticEmbeddingInstance(self._name, table)
			self._loaded[normalizer.name] = loaded

			if path and not path.exists():
				loaded.save(path)

			loaded.free_table()

		return loaded


# https://github.com/avidale/compress-fasttext
#class CompressedStaticEmbedding:
#	def create_instance(self, normalizer):
#		compress_fasttext.models.CompressedFastTextKeyedVectors.load


class StaticEmbeddingFromFile(StaticEmbedding):
	def __init__(self, embeddings=None, tokens=None, vectors=None, name=None):
		if embeddings:
			self._tokens = list(embeddings.keys())
			self._vectors = np.vstack(embeddings.values())
			self._path = None

	@staticmethod
	def import_from_file(path, **kwargs):
		tokens, vectors = _load_glove_txt(path)
		return StaticEmbedding(tokens=tokens, vectors=vectors, **kwargs)

	def _load(self):
		raise NotImplementedError()


class EmbeddingLoadingProgress:
	def __init__(self, pbar):
		self._pbar = pbar
		self._last = 0

	def update(self, ratio):
		d = ratio - self._last
		self._last = ratio
		self._pbar.update(d)


class StaticEmbeddingInstance:
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
			compression='snappy',
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


class Glove(StaticEmbedding):
	def __init__(self, name="6B"):
		"""
		:param name: one of "6B", "42B.300d", "840B.300d",
		"twitter.27B", see https://nlp.stanford.edu/projects/glove/
		"""

		self._glove_name = name

		self._base_path = Path.home() / ".vectorian" / "embeddings" / "glove"
		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{name}.parquet"

		super().__init__(path=pq_path, name=f"glove-{name}")

	def _load(self):
		name = self._glove_name
		txt_data_path = self._base_path / name

		if not txt_data_path.exists():
			url = f"http://nlp.stanford.edu/data/glove.{name}.zip"
			download.download(url, txt_data_path, kind="zip", progressbar=True)

		return _load_glove_txt(
			txt_data_path / f"glove.{name}.300d.txt")


class FastText(StaticEmbedding):
	def __init__(self, lang):
		"""
		:param lang: language code of fasttext encodings, see
		https://fasttext.cc/docs/en/crawl-vectors.html
		"""

		self._lang = lang
		self._base_path = Path.home() / ".vectorian" / "embeddings" / "fasttext"
		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{lang}.parquet"

		super().__init__(path=pq_path, name=f"fasttext-{self._lang}")

	def _load(self):
		import fasttext.util
		lang = self._lang

		os.chdir(self._base_path)
		filename = fasttext.util.download_model(lang, if_exists='ignore')
		ft = fasttext.load_model(filename)

		words = ft.get_words()
		ndims = ft.get_dimension()
		embeddings = np.empty((len(words), ndims), dtype=np.float32)
		for i, w in enumerate(tqdm(words, desc=f"Importing fasttext ({lang})")):
			embeddings[i] = ft.get_word_vector(w)
		return words, embeddings


class ContextualEmbedding:
	def encode(self, doc):
		raise NotImplementedError()


class TransformerEmbedding(ContextualEmbedding):
	def encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		return doc._.trf_data.tensors[-1]
