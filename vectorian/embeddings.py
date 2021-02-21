import vectorian.core as core

from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import download


def _make_table(tokens, embeddings):
	vecs = [pa.array(embeddings[:, i]) for i in range(embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(embeddings.shape[1])]

	return pa.Table.from_arrays(
		[pa.array(tokens, type=pa.string())] + vecs,
		['token'] + vecs_name)


class Embedding:
	pass


def _load_fasttext_txt(csv_path):
	tokens = []
	with open(csv_path, "r") as f:
		n_rows, n_cols = map(int, f.readline().strip().split())

		embeddings = np.empty(
			shape=(n_rows, n_cols), dtype=np.float64)

		for _ in tqdm(range(n_rows)):
			values = f.readline().strip().split()
			if values:
				t = values[0]
				if t and t.isalpha() and t.lower() == t:
					embeddings[len(tokens), :] = values[1:]
					tokens.append(t)

	embeddings = embeddings[:len(tokens), :]
	embeddings = embeddings.astype(np.float32)

	return tokens, embeddings


def _load_glove_txt(csv_path):
	tokens = []

	print(f"Loading {csv_path}")
	with open(csv_path, "r") as f:
		text = f.read()

	lines = text.split("\n")
	n_rows = len(lines)
	n_cols = len(lines[0].strip().split()) - 1

	embeddings = np.empty(
		shape=(n_rows, n_cols), dtype=np.float64)

	for line in tqdm(lines):
		values = line.strip().split()
		if values:
			t = values[0]
			if t and t.isalpha() and t.lower() == t:
				embeddings[len(tokens), :] = values[1:]
				tokens.append(t)

	embeddings = embeddings[:len(tokens), :]
	embeddings = embeddings.astype(np.float32)

	return tokens, embeddings


class Glove(Embedding):
	def __init__(self, name="6B"):
		"""
		:param name: one of "6B", "42B.300d", "840B.300d",
		"twitter.27B", see https://nlp.stanford.edu/projects/glove/
		"""

		self._name = name

		self._base_path = Path.home() / ".vectorian" / "embeddings" / "glove"
		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{name}.parquet"

		if not pq_path.exists():
			txt_data_path = self._base_path / name

			if not txt_data_path.exists():
				url = f"http://nlp.stanford.edu/data/glove.{name}.zip"
				download.download(url, txt_data_path, kind="zip", progressbar=True)

			tokens, embeddings = _load_glove_txt(
				txt_data_path / f"glove.{name}.300d.txt")

			pq.write_table(
				_make_table(tokens, embeddings),
				pq_path,
				compression='snappy',
				version='2.0')

		self._table = pq.read_table(pq_path)

	@property
	def name(self):
		return f"glove-{self._name}"

	@lru_cache(1)
	def to_core(self):
		embedding = core.FastEmbedding(self.name, self._table)
		self._table = None  # free up memory
		return embedding


class FastText(Embedding):
	def __init__(self, lang):
		"""
		:param lang: language code of fasttext encodings, see
		https://fasttext.cc/docs/en/crawl-vectors.html
		"""

		self._lang = lang
		self._base_path = Path.home() / ".vectorian" / "embeddings" / "fasttext"
		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{lang}.parquet"

		if not pq_path.exists():
			tokens, embeddings = self._load()
			embeddings = embeddings.astype(np.float32)

			pq.write_table(
				_make_table(tokens, embeddings),
				pq_path,
				compression='snappy',
				version='2.0')

		self._table = pq.read_table(pq_path)

	def _load(self):
		import fasttext.util
		lang = self._lang

		os.chdir(self._base_path)
		filename = fasttext.util.download_model(lang, if_exists='ignore')
		ft = fasttext.load_model(filename)

		words = ft.get_words()
		ndims = ft.get_dimension()
		embeddings = np.empty((len(words), ndims), dtype=np.float64)
		for i, w in enumerate(tqdm(words)):
			embeddings[i] = ft.get_word_vector(w)
		return words, embeddings

	@property
	def name(self):
		return f"fasttext-{self._lang}"

	@lru_cache(1)
	def to_core(self):
		embedding = core.FastEmbedding(self.name, self._table)
		self._table = None  # free up memory
		return embedding
