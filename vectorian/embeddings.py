import vectorian.core as core

from tqdm import tqdm
from sklearn.preprocessing import normalize
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os


def _make_table(tokens, embeddings):
	vecs = [pa.array(embeddings[:, i]) for i in range(embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(embeddings.shape[1])]

	return pa.Table.from_arrays(
		[pa.array(tokens, type=pa.string())] + vecs,
		['token'] + vecs_name)


'''
class Precomputed:
	def __init__(self, name, csv_path):
		self._base_path = Path.home() / ".vectorian" / "embeddings" / "csv"

		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{name}.parquet"

		if not pq_path.exists():
			tokens, embeddings = self._load(csv_path)

			embeddings = normalize(embeddings, axis=1, norm='l2')
			embeddings = embeddings.astype(np.float32)

			pq.write_table(
				_make_table(tokens, embeddings),
				pq_path,
				compression='snappy',
				version='2.0')

		self._table = pq.read_table(pq_path)

	def _load(self, csv_path):
		tokens = []
		with open(csv_path, "r") as f:
			n_rows, n_cols = map(int, f.readline().strip().split())

			embeddings = np.empty(
				shape=(n_rows, n_cols), dtype=np.float64)

			for _ in tqdm(range(n_rows)):
				values = f.readline().strip().split()

				t = values[0]
				if t.isalpha() and t.lower() == t:
					embeddings[len(tokens), :] = values[1:]

					tokens.append(t)

		embeddings = embeddings[:len(tokens), :]

		return tokens, embeddings
'''


class FastText:
	def __init__(self, lang):
		self._lang = lang
		self._base_path = Path.home() / ".vectorian" / "embeddings" / "fasttext"

		self._base_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._base_path / f"{lang}.parquet"

		if not pq_path.exists():
			tokens, embeddings = self._load()

			embeddings = normalize(embeddings, axis=1, norm='l2')
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

	def to_core(self):
		return core.FastEmbedding(self.name, self._table)
