import vectorian.core as core
import multiprocessing
import multiprocessing.pool
import time
import numpy as np
import bisect
import json
import yaml
import collections
import contextlib

from cached_property import cached_property
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path
from vectorian.corpus.document import TokenTable, InternalMemoryText
from vectorian.embeddings import Vectors, ProxyVectorsRef


class Query:
	def __init__(self, index, vocab, text, options):
		self._index = index
		self._vocab = vocab
		self._text = text
		self._options = options

	@property
	def index(self):
		return self._index

	@property
	def text(self):
		return self._text

	@property
	def options(self):
		return self._options

	def prepare(self, nlp):
		return PreparedQuery(self, self._vocab, nlp)


class PreparedQuery:
	def __init__(self, query, vocab, nlp):
		self._query = query
		self._vocab = vocab

		doc = nlp(self.text_str)

		# FIXME gather contextual_embeddings actually used in this query
		# by analyzing query

		contextual_embeddings = collections.defaultdict(list)
		for e in self._query.index.session.embeddings:
			if e.is_contextual:
				contextual_embeddings[e.name].append(e.encode(doc))

		contextual_embeddings = dict(
			(k, np.vstack(v)) for k, v in contextual_embeddings.items())

		tokens = doc.to_json()["tokens"]

		for token_attr in ('pos', 'tag'):  # FIXME token_filter
			mask = self._mask(tokens, f'{token_attr}_filter', token_attr)
			if mask is not None:
				tokens = [t for t, m in zip(tokens, mask) if m]
				contextual_embeddings = dict(
					(k, v[mask, :]) for k, v in contextual_embeddings.items())

		token_mask = np.zeros((len(tokens),), dtype=np.bool)
		token_table = TokenTable(self.text_str, self.index.session.normalizers)
		for i, t in enumerate(tokens):
			token_mask[i] = token_table.add(t)

		contextual_embeddings = dict(
			(k, v[token_mask, :]) for k, v in contextual_embeddings.items())

		self._contextual_embeddings = dict(
			(k, ProxyVectorsRef(Vectors(v))) for k, v in contextual_embeddings.items())

		query = core.Query(
			self.index,
			self._vocab,
			self._contextual_embeddings)
		query.initialize(
			token_table.to_dict(),
			**self._query.options)

		self._compiled = query
		self._tokens = self._compiled.tokens

	@property
	def index(self):
		return self._query.index

	@property
	def text_str(self):
		return self._query.text

	@contextlib.contextmanager
	def text(self):
		yield InternalMemoryText(self.text_str)

	@property
	def options(self):
		return self._query.options

	@property
	def n_tokens(self):
		return self._compiled.n_tokens

	@cached_property
	def span(self):
		from vectorian.corpus.document import Span
		return Span(self, self._tokens, 0, self.n_tokens)

	def __getitem__(self, i):
		return self.span[i]

	def __len__(self):
		return len(self.span)

	def _mask(self, tokens, name, k):
		f = self._query.options.get(name, None)
		if f:
			s = set(f)
			return np.array([t[k] not in s for t in tokens], dtype=np.bool)
		else:
			return None

	@property
	def compiled(self):
		return self._compiled


Region = namedtuple('Region', [
	's', 'match', 'gap_penalty'])


TokenMatch = namedtuple('TokenMatch', [
	'pos_s', 'edges'])


TokenMatchEdge = namedtuple('TokenMatchEdge', [
	't', 'pos_t', 'flow', 'distance', 'metric'])


PartitionData = namedtuple('PartitionData', [
	'level', 'window_size', 'window_step'])


class Match:
	@property
	def doc_span(self):
		return self.document.span(self.query.index.partition, self.slice_id)

	@property
	def query(self):
		raise NotImplementedError()

	@property
	def document(self):
		raise NotImplementedError()

	@property
	def slice_id(self):
		raise NotImplementedError()

	@property
	def score(self):
		raise NotImplementedError()

	@property
	def metric(self):
		raise NotImplementedError()

	@property
	def omitted(self):
		raise NotImplementedError()

	@property
	def regions(self):
		raise NotImplementedError()

	@property
	def level(self):
		raise NotImplementedError()

	@property
	def flow(self):
		return None

	def to_json(self, context_size=10):
		regions = []

		doc = self.document
		partition = self.query.options["partition"]

		span_info = doc.span_info(
			PartitionData(**partition), self.slice_id)

		for region in self.regions(context_size):
			s = region.s
			if region.match:
				edges = []
				for e in region.match.edges:
					edges.append({
						't': e.t,
						'pos_t': e.pos_t,
						'flow': e.flow,
						'distance': e.distance,
						'metric': e.metric
					})

				regions.append(dict(
					s=s,
					pos_s=region.match.pos_s,
					edges=edges))
			else:
				regions.append(dict(
					s=s,
					gap_penalty=region.gap_penalty))

		data = dict(
			slice=self.slice_id,
			location=span_info,
			score=self.score,
			metric=self.metric,
			regions=regions,
			omitted=self.omitted,
			level=self.level)

		return data


class CoreMatch(Match):
	def __init__(self, session, query, c_match):
		self._session = session
		self._query = query
		self._c_match = c_match
		self._level = "word"

	@property
	def query(self):
		return self._query

	@property
	def document(self):
		return self._session.documents[self._c_match.document.id]

	@property
	def slice_id(self):
		return self._c_match.slice_id

	@property
	def score(self):
		return self._c_match.score

	@property
	def metric(self):
		return self._c_match.metric

	@property
	def omitted(self):
		t_text = self._query.text_str
		omitted = [t_text[slice(*s)] for s in self._c_match.omitted]
		return omitted

	def regions(self, context_size=10):
		with self.document.text() as s_text_st:
			s_text = s_text_st.get()
			t_text = self.query.text_str

			regions = []
			for r in self._c_match.regions(context_size):
				if r.matched:
					edges = []
					for i in range(r.num_edges):
						edges.append(TokenMatchEdge(
							t=t_text[slice(*r.t(i))],
							pos_t=r.pos_t(i),
							flow=r.flow(i),
							distance=r.distance(i),
							metric=r.metric(i)))

					regions.append(Region(
						s=s_text[slice(*r.s)],
						match=TokenMatch(
							pos_s=r.pos_s,
							edges=edges),
						gap_penalty=r.mismatch_penalty))
				else:
					regions.append(Region(
						s=s_text[slice(*r.s)],
						match=None,
						gap_penalty=r.mismatch_penalty))

		return regions

	@property
	def level(self):
		return self._level

	@property
	def flow(self):
		return self._c_match.flow


class PyMatch:
	def __init__(self, query, document, slice_id, score, metric=None, omitted=None, regions=None, level="word"):
		self._query = query
		self._document = document
		self._slice_id = slice_id
		self._score = score
		self._metric = metric or ""
		self._omitted = omitted or []
		self._regions = regions or []
		self._level = level

	@property
	def query(self):
		return self._query

	@property
	def document(self):
		return self._document

	@property
	def slice_id(self):
		return self._slice_id

	@property
	def score(self):
		return self._score

	@property
	def metric(self):
		return self._metric

	@property
	def omitted(self):
		return self._omitted

	@property
	def regions(self):
		return self._regions

	@property
	def level(self):
		return self._level


class Index:
	def __init__(self, partition, metric):
		self._partition = partition
		self._metric = metric

	@property
	def partition(self):
		return self._partition

	@property
	def session(self):
		return self._partition.session

	@property
	def metric(self):
		return self._metric

	def describe(self):
		data = {
			'partition': self._partition.to_args(),
			'metric': self._metric.to_args(self)
		}
		print(yaml.dump(data))

	def find(
		self, text,
		n=10, min_score=0.0, debug=None,
		run_task=None, make_result=None,
		options: dict = dict()):

		options = options.copy()

		options["max_matches"] = n
		options["min_score"] = min_score
		if debug is not None:
			options["debug"] = debug
		options["partition"] = self._partition.to_args()

		metric_args = self._metric.to_args(self)
		if metric_args:
			options["metric"] = metric_args

		start_time = time.time()

		session = self._partition.session
		if make_result is None:
			make_result = session.make_result
		if run_task is None:
			run_task = session.on_progress

		query = Query(self, session.vocab, text, options)
		matches = run_task(lambda progress: self._find(query, progress=progress))

		return make_result(
			self,
			matches,
			duration=time.time() - start_time)


class BruteForceIndex(Index):
	def __init__(self, *args, nlp, **kwargs):
		super().__init__(*args, **kwargs)
		self._nlp = nlp

	def _find(self, query, n_threads=None, progress=None):
		p_query = query.prepare(self._nlp)

		if len(p_query) == 0:
			return []

		c_query = p_query.compiled

		def find_in_doc(x):
			return x, x.find(c_query)

		docs = self.session.c_documents

		total = sum([x.n_tokens for x in docs])
		done = 0

		if n_threads is None:
			n_threads = min(len(docs), multiprocessing.cpu_count())

		results = None
		with multiprocessing.pool.ThreadPool(processes=n_threads) as pool:
			for doc, r in pool.imap_unordered(find_in_doc, docs):
				if results is None:
					results = r
				else:
					results.extend(r)
				done += doc.n_tokens
				if progress:
					progress(done / total)

		session = self.session
		return [CoreMatch(session, p_query, m) for m in results.best_n(-1)]


def chunks(x, n):
	for i in range(0, len(x), n):
		yield x[i:i + n]


# the next three functions are taken from:
# https://gist.github.com/mdouze/e4bdb404dbd976c83fe447e529e5c9dc
# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5

def get_phi(xb):
	return (xb ** 2).sum(1).max()


def augment_xb(xb, phi=None):
	norms = (xb ** 2).sum(1)
	if phi is None:
		phi = norms.max()
	extracol = np.sqrt(phi - norms)
	return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq):
	extracol = np.zeros(len(xq), dtype=np.float32)
	return np.hstack((xq, extracol.reshape(-1, 1)))


class SentenceEmbeddingIndex(Index):
	def __init__(self, partition, metric, encoder, vectors=None, faiss_description='Flat'):
		super().__init__(partition, metric)

		import faiss

		self._partition = partition
		self._metric = metric
		self._encoder = encoder

		session = self._partition.session

		doc_starts = [0]
		for i, doc in enumerate(session.documents):
			doc_starts.append(doc.n_spans(self._partition))
		self._doc_starts = np.cumsum(np.array(doc_starts, dtype=np.int32))

		if vectors is not None:
			corpus_vec = vectors
		else:
			corpus_vec = []

			chunk_size = 50
			n_spans = self._doc_starts[-1]

			with tqdm(desc="Encoding", total=n_spans) as pbar:
				for i, doc in enumerate(session.documents):
					spans = list(doc.spans(self._partition))
					for chunk in chunks(spans, chunk_size):
						doc_vec = encoder(chunk)
						corpus_vec.append(doc_vec)
						pbar.update(len(chunk))

			corpus_vec = np.vstack(corpus_vec)
			corpus_vec /= np.linalg.norm(corpus_vec, axis=1, keepdims=True)

		# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
		# https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
		# https://github.com/facebookresearch/faiss/wiki/The-index-factory
		# e.g. ""Flat", "PCA128,Flat", "LSH"

		self._ip_to_l2 = faiss_description.split(",")[-1] != "Flat"

		if self._ip_to_l2:
			corpus_vec = augment_xb(corpus_vec)
			metric = faiss.METRIC_L2
		else:
			metric = faiss.METRIC_INNER_PRODUCT

		n_dims = corpus_vec.shape[-1]
		index = faiss.index_factory(n_dims, faiss_description, metric)
		index.train(corpus_vec)
		index.add(corpus_vec)

		self._index = index
		self._corpus_vec = corpus_vec

	@property
	def session(self):
		return self._partition.session

	@staticmethod
	def load(session, metric, encoder, path):
		path = Path(path)
		corpus_vec = []
		with open(path / "index.json", "r") as f:
			data = json.loads(f.read())
		if data["metric"] != "sentence_embedding":
			raise RuntimeError(f"index at {path} is not a SentenceEmbedding index")
		for i, doc in enumerate(session.documents):
			p = path / (doc.caching_name + ".npy")
			if not p.exists():
				raise FileNotFoundError(p)
			corpus_vec.append(np.load(p))
		corpus_vec = np.vstack(corpus_vec)
		return SentenceEmbeddingIndex(
			session.partition(**data["partition"]),
			metric, encoder, corpus_vec)

	def save(self, path):
		path = Path(path)
		path.mkdir(exist_ok=True)

		offset = 0
		session = self._partition.session

		for doc in tqdm(session.documents, desc="Saving"):
			size = doc.n_spans(self._partition)
			np.save(
				str(path / (doc.caching_name + ".npy")),
				self._corpus_vec[offset:size],
				allow_pickle=False)
			offset += size

		with open(path / "index.json", "w") as f:
			f.write(json.dumps({
				'metric': 'sentence_embedding',
				'partition': self._partition.to_args()
			}))

	def _find(self, query, progress=None):
		query_vec = self._encoder([query.text])
		query_vec /= np.linalg.norm(query_vec)

		if self._ip_to_l2:
			query_vec = augment_xq(query_vec)

		distance, index = self._index.search(
			query_vec, query.options["max_matches"])

		matches = []
		for d, i in zip(distance[0], index[0]):
			if i < 0:
				break

			doc_index = bisect.bisect_left(self._doc_starts, i)
			if doc_index > 0 and self._doc_starts[doc_index] > i:
				doc_index -= 1
			sent_index = i - self._doc_starts[doc_index]

			#print(i, doc_index, sent_index, self._doc_starts)

			doc = self.session.documents[doc_index]
			score = (d + 1) * 0.5

			#print(c_doc, sentence_id)
			#print(c_doc.sentence(sentence_id))
			#print(score, d)

			span_text = doc.span(self._partition, sent_index)
			#print(sent_text, len(sent_text), c_doc.sentence_info(sent_index))

			regions = [Region(
				s=span_text.strip(),
				match=None, gap_penalty=0)]

			matches.append(PyMatch(
				query,
				doc,
				sent_index,
				score,
				self._metric.name,
				regions=regions,
				level="span"
			))

		return matches
