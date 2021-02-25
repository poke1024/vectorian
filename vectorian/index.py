import vectorian.core as core
import spacy
import multiprocessing
import multiprocessing.pool
import time
import numpy as np
import bisect

from collections import namedtuple
from tqdm import tqdm
from vectorian.corpus.document import TokenTable


class Query:
	def __init__(self, vocab, doc, options):
		self._vocab = vocab
		self._doc = doc
		self._options = options

	@property
	def text(self):
		return self._doc.text

	@property
	def options(self):
		return self._options

	def _filter(self, tokens, name, k):
		f = self._options.get(name, None)
		if f:
			s = set(f)
			return [t for t in tokens if t[k] not in s]
		else:
			return tokens

	def to_core(self):
		tokens = self._doc.to_json()["tokens"]
		tokens = self._filter(tokens, 'pos_filter', 'pos')
		tokens = self._filter(tokens, 'tag_filter', 'tag')

		token_table = TokenTable()
		token_table.extend(self._doc.text, tokens)

		return core.Query(
			self._vocab,
			self._doc.text,
			token_table.to_arrow(),
			**self._options)


Region = namedtuple('Region', [
	's', 'match', 'gap_penalty'])


RegionMatch = namedtuple('TokenMatch', [
	't', 'pos_s', 'pos_t', 'similarity', 'weight', 'metric'])


class Match:
	def __init__(self, document, sentence, score, metric=None, omitted=None, regions=None):
		self._document = document
		self._sentence = sentence
		self._score = score
		self._metric = metric or ""
		self._omitted = omitted or []
		self._regions = regions or []

	@staticmethod
	def from_core(c_match):
		regions = []
		for r in c_match.regions:
			if r.matched:
				regions.append(Region(
					s=r.s.decode('utf-8', errors='ignore'),
					pos_s=r.pos_s.decode('utf-8', errors='ignore'),
					match=TokenMatch(
						t=r.t.decode('utf-8', errors='ignore'),
						pos_t=r.pos_t.decode('utf-8', errors='ignore'),
						similarity=r.similarity,
						weight=r.weight,
						metric=r.metric.decode('utf-8', errors='ignore')),
					gap_penalty=r.mismatch_penalty))
			else:
				regions.append(Region(
					s=r.s.decode('utf-8', errors='ignore'),
					pos_s=None,
					match=None,
					gap_penalty=r.mismatch_penalty))

		return Match(
			c_match.document, c_match.sentence, c_match.score,
			c_match.metric, c_match.omitted, regions)

	@property
	def document(self):
		return self._document

	@property
	def sentence(self):
		return self._sentence

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


class Index:
	def __init__(self, session, metric):
		self._session = session
		self._metric = metric

	def find(
		self, doc: spacy.tokens.doc.Doc,
		n=100, min_score=0.2,
		options: dict = dict()):

		if not isinstance(doc, spacy.tokens.doc.Doc):
			raise TypeError("please specify a spaCy document as query")

		metric_args = self._metric.to_args(self._session)

		options = options.copy()
		if metric_args:
			options["metric"] = metric_args
		options["max_matches"] = n
		options["min_score"] = min_score

		start_time = time.time()

		query = Query(self._session.vocab, doc, options)
		result_class, matches = self._session.run_query(self._find, query)

		return result_class(
			matches,
			duration=time.time() - start_time)


class BruteForceIndex(Index):
	def _find(self, query, n_threads=None, progress=None):
		c_query = query.to_core()

		def find_in_doc(x):
			return x, x.find(c_query)

		docs = self._session.documents

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

		return [Match.from_core(m) for m in results.best_n(-1)]


class SentenceEmbeddingIndex(Index):
	def __init__(self, session, metric, encoder):
		super().__init__(session, metric)

		self._encoder = encoder
		self._metric = metric

		import faiss

		corpus_vec = []
		doc_starts = [0]
		for i, doc in enumerate(tqdm(session.documents, "encoding")):
			sents = doc.sentences
			doc_vec = encoder(sents)
			n_dims = doc_vec.shape[-1]
			corpus_vec.append(doc_vec)
			doc_starts.append(len(sents))

		corpus_vec = np.vstack(corpus_vec)
		corpus_vec /= np.linalg.norm(corpus_vec, axis=1, keepdims=True)

		for v in corpus_vec:
			print("?", np.linalg.norm(v))

		self._doc_starts = np.cumsum(np.array(doc_starts, dtype=np.int32))

		# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
		# https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

		#n_bits = 2 * n_dims
		#index = faiss.IndexLSH(n_dims, n_bits)

		pca_dim = 128
		if corpus_vec.shape[0] < pca_dim:
			pca_dim = None

		# https://github.com/facebookresearch/faiss/wiki/The-index-factory
		index = faiss.index_factory(n_dims, "Flat", faiss.METRIC_INNER_PRODUCT)
		#index = faiss.index_factory(n_dims, "PCA128,LSH", faiss.METRIC_INNER_PRODUCT)
		#index = faiss.index_factory(n_dims, "LSH", faiss.METRIC_INNER_PRODUCT)
		index.train(corpus_vec)
		index.add(corpus_vec)

		self._index = index

	def _find(self, query, progress=None):
		query_vec = self._encoder([query.text])
		query_vec /= np.linalg.norm(query_vec)

		distance, index = self._index.search(
			query_vec, query.options["max_matches"])

		matches = []
		for d, i in zip(distance[0], index[0]):
			if i < 0:
				break

			doc_index = bisect.bisect_left(self._doc_starts, i)
			sent_index = i - self._doc_starts[doc_index]

			#print(i, doc_index, self._doc_starts)

			c_doc = self._session.documents[doc_index]
			score = 1 - (d + 1) * 0.5

			#print(c_doc, sentence_id)
			#print(c_doc.sentence(sentence_id))
			print(score, d)

			sents = c_doc.sentences  # FIXME
			regions = [Region(
				s=sents[sent_index], match=None, gap_penalty=0)]

			matches.append(Match(
				c_doc,
				sent_index,
				score,
				self._metric.name,
				regions=regions
			))

		print(matches)
		return matches
