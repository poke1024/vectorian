import vectorian.core as core

from tqdm import tqdm
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import download
import logging


def _make_table(tokens, embeddings, normalizer, strategy='select'):
	embeddings = embeddings.astype(np.float32)

	with open("/Users/arbeit/debug.txt", "a") as f:
		f.write("stage1\n")
		f.write(str(embeddings[tokens.index("speak")]) + "\n")

	f_mask = np.zeros((embeddings.shape[0],), dtype=np.bool)
	f_tokens = []
	token_to_ids = dict()

	for i, t in enumerate(tqdm(tokens, desc="Normalizing Tokens")):
		nt = normalizer(t)
		if nt is None:
			continue
		if strategy != 'merge' and nt != t:
			continue
		indices = token_to_ids.get(nt)
		if indices is None:
			token_to_ids[nt] = [i]
			f_tokens.append(nt)
			f_mask[i] = True
		else:
			indices.append(i)

	if strategy == 'merge':
		for indices in tqdm(token_to_ids.values(), desc="Merging Tokens", total=len(token_to_ids)):
			if len(indices) > 1:
				i = indices[0]
				embeddings[i] = np.mean(embeddings[indices], axis=0)

	f_embeddings = embeddings[f_mask]
	embeddings = None

	assert f_embeddings.shape[0] == len(f_tokens)

	with open("/Users/arbeit/debug.txt", "a") as f:
		f.write("stagex\n")
		f.write(str(token_to_ids["speak"]) + "\n")
		for x in token_to_ids["speak"]:
			f.write(tokens[x] + "\n")
		f.write("stage2\n")
		f.write(str(f_embeddings[f_tokens.index("speak")]) + "\n")

	vecs = [pa.array(f_embeddings[:, i]) for i in range(f_embeddings.shape[1])]
	vecs_name = [('v%d' % i) for i in range(f_embeddings.shape[1])]
	f_embeddings = None

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


class StaticEmbedding:
	def __init__(self):
		self._loaded = {}
		cache_path = Path.home() / ".vectorian" / "embeddings"
		cache_path.mkdir(exist_ok=True, parents=True)
		self._cache_path = cache_path

	def _load(self):
		raise NotImplementedError()

	@property
	def name(self):  # i.e. display name
		return self.unique_name

	@property
	def unique_name(self):
		raise NotImplementedError()

	def create_instance(self, normalizer, strategy):
		loaded = self._loaded.get(normalizer.name)
		if loaded is None:
			name = self.unique_name

			normalized_cache_path = self._cache_path / 'pa
			normalized_cache_path.mkdir(exist_ok=True, parents=True)
			pq_path = normalized_cache_path / f"{name}-{normalizer.name}-{strategy}.parquet"

			if pq_path.exists():
				with tqdm(desc="Opening " + self.name, total=1,  bar_format='{l_bar}{bar}') as pbar:
					table = pq.read_table(pq_path, memory_map=True)
					pbar.update(1)
			else:
				tokens, vectors = self._load()
				table = _make_table(
					tokens, vectors, normalizer.unpack(), strategy)

			loaded = StaticEmbeddingInstance(name, table)
			self._loaded[normalizer.name] = loaded

			if pq_path and not pq_path.exists():
				loaded.save(pq_path)

			loaded.free_table()

		return loaded


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


class Glove(StaticEmbedding):
	def __init__(self, name="6B"):
		"""
		:param name: one of "6B", "42B.300d", "840B.300d",
		"twitter.27B", see https://nlp.stanford.edu/projects/glove/
		"""

		self._glove_name = name

		self._cache_path = Path.home() / ".vectorian" / "embeddings" / "glove"
		self._cache_path.mkdir(exist_ok=True, parents=True)
		pq_path = self._cache_path / f"{name}.parquet"

		super().__init__(path=pq_path, name=f"glove-{name}")

	def _load(self):
		name = self._glove_name
		txt_data_path = self._cache_path / name

		if not txt_data_path.exists():
			url = f"http://nlp.stanford.edu/data/glove.{name}.zip"
			download.download(url, txt_data_path, kind="zip", progressbar=True)

		return _load_glove_txt(
			txt_data_path / f"glove.{name}.300d.txt")


class FastTextVectors(StaticEmbedding):
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


class CompressedFastTextVectors(StaticEmbedding):
	def _load(self):
		import compress_fasttext
		small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(self._path)


class FacebookFastTextVectors(StaticEmbedding):
	def __init__(self, lang):
		"""
		:param lang: language code of precomputed fasttext encodings, see
		https://fasttext.cc/docs/en/crawl-vectors.html
		"""

		super().__init__()
		self._lang = lang

	def _load(self):
		import fasttext
		import fasttext.util
		download_path = self._cache_path / 'models'
		download_path.mkdir(exist_ok=True, parents=True)
		os.chdir(download_path)
		with tqdm(desc="Downloading " + self.name, total=1, bar_format='{l_bar}{bar}') as pbar:
			filename = fasttext.util.download_model(
				self._lang, if_exists='ignore')
			pbar.update(1)
		with tqdm(desc="Opening " + self.name, total=1, bar_format='{l_bar}{bar}') as pbar:
			ft = fasttext.load_model(str(download_path / filename))
			pbar.update(1)
		words = ft.get_words()
		ndims = ft.get_dimension()
		embeddings = np.empty((len(words), ndims), dtype=np.float32)
		for i, w in enumerate(tqdm(words, desc=f"Importing {self.unique_name}")):
			embeddings[i] = ft.get_word_vector(w)
		return words, embeddings

	@property
	def unique_name(self):
		return f"fasttext-{self._lang}"


'''
class FastText(StaticEmbedding):
	def __init__(self, loader):
		self._custom_model_path = None

		if lang:
			unique_name = f"fasttext-{self._lang}"
		else:
			self._custom_model_path = Path(path)
			if unique_name is None:
				unique_name = self._custom_model_path.stem

		self._cache_path = Path.home() / ".vectorian" / "embeddings"
		self._cache_path.mkdir(exist_ok=True, parents=True)

		unique_name = loader.name()
		pq_path = self._cache_path / f"{unique_name}.parquet"

		super().__init__(path=pq_path, name=unique_name)

	def _model_path(self):
		if self._custom_model_path is None:
			import fasttext.util
			os.chdir(self._cache_path)
			filename = fasttext.util.download_model(
				self._lang, if_exists='ignore')
			return self._cache_path / filename
		else:
			return self._custom_model_path

'''


class ContextualEmbedding:
	def encode(self, doc):
		raise NotImplementedError()


class TransformerEmbedding(ContextualEmbedding):
	def encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		return doc._.trf_data.tensors[-1]
