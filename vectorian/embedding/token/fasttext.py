import re
import json
import io
import os
import numpy as np
import contextlib
import vectorian.core as core
from pathlib import Path
from vectorian.tqdm import tqdm
from .keyed import StaticEmbedding, StaticEmbeddingEncoder, KeyedVectors
from ..utils import make_cache_path, extraction_tqdm
from ..vectors import Vectors


class CompressedFastTextVectors(StaticEmbedding):
	def __init__(self, path):
		import compress_fasttext

		self._name = Path(path).stem
		self._wv = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(path))

	def create_encoder(self, session):
		return KeyedVectors.Encoder(self, self._name, self._wv)

	@property
	def name(self):
		return self._name


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
	class Encoder(StaticEmbeddingEncoder):
		def __init__(self, embedding, name, ft):
			self._embedding = embedding
			self._name = name
			self._ft = ft

		@property
		def embedding(self):
			return self._embedding

		@property
		def name(self):
			return self._name

		def word_vec(self, t):
			return self._ft.get_word_vector(t)

		@property
		def dimension(self):
			return self._ft.get_dimension()

		def encode_tokens(self, tokens):
			data = np.empty((len(tokens), self.dimension), dtype=np.float32)
			for i, t in enumerate(extraction_tqdm(tokens, self.name)):
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

		download_path = make_cache_path() / 'models'
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

	def create_encoder(self, session):
		return PretrainedFastText.Encoder(
			self, self.name, self._ft)

	@property
	def name(self):
		return f"fasttext-{self._lang}"