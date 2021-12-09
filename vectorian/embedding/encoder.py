import numpy as np
import json
import h5py
import cachetools
import sqlite3
import uuid

from pathlib import Path
from .vectors import Vectors
from vectorian.tqdm import tqdm


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


class SpanEncoder:
	def __init__(self, partition):
		self._partition = partition

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def embedding(self):
		raise NotImplementedError()

	def encode(self, docs, pbar=False):
		raise NotImplementedError()


class InMemorySpanEncoder(SpanEncoder):
	def __init__(self, partition, span_embedding):
		super().__init__(partition)
		self._embedding = span_embedding

	def vector_size(self, session):
		return self._embedding.vector_size(session)

	@property
	def embedding(self):  # i.e. token embedding
		return self._embedding.embedding

	def encode(self, docs, pbar=False):
		partition = self._partition

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

		for i, v in enumerate(self._embedding.encode(partition.session, gen_spans())):
			out[i_spans[i]:i_spans[i + 1], :] = v

		return Vectors(out)

	def to_cached(self, cache_size=150):
		return CachedSpanEncoder(self._partition, self._embedding, cache_size)


class CachedSpanEncoder(SpanEncoder):
	def __init__(self, partition, span_embedding, cache_size=150):
		super().__init__(partition)

		self._corpus = partition.session.corpus
		self._encoder = InMemorySpanEncoder(partition, span_embedding)
		self._embedding = span_embedding
		self._cache = cachetools.LRUCache(cache_size)
		self._uuid = self._gen_uuid()

	def _gen_uuid(self):
		conf_key = json.dumps({
			'embedding': self._embedding.unique_id,
			'partition': self._partition.cache_key
		})


		return unique_id

	def _emb_cache_path(self, doc):
		p = self._corpus.get_doc_path(doc) / "span_embeddings" / (self._uuid + ".h5")
		if not p.exists():
			return None


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

	def _mk_cache_key(self, doc):
		if doc.corpus is None:
			return None
		uid = doc.corpus_id
		if uid is None:
			return None
		return uid

	def _store(self, doc, v_doc):
		cache_key = self._mk_cache_key(doc)
		if cache_key is not None:
			self._cache[cache_key] = v_doc
			Vectors(v_doc).save(emb_path / unique_id)

	def encode(self, docs, pbar=False):
		partition = self._partition

		n_spans = [doc.n_spans(partition) for doc in docs]
		i_spans = np.cumsum([0] + n_spans)

		out = np.empty((sum(n_spans), self.vector_size(partition.session)))

		new = []
		index = []

		# we assume all docs stem from the same corpus. otherwise our caching
		# ids would not be reliable.
		for doc in docs:
			if doc.corpus not in (None, self._corpus):
				raise RuntimeError(f"doc {doc} has corpus {doc.corpus}, expected either None or {self._corpus}")

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

				self._store(new[i], v_doc)

		return Vectors(out)

	def to_cached(self, cache_size=None):
		return self
