import vectorian.core as core
import spacy
import multiprocessing
import multiprocessing.pool

from vectorian.corpus.document import TokenTable
from vectorian.corpus import Corpus
from vectorian.corpus import Document



class Query:
	def __init__(self, vocab, doc, options):
		self._vocab = vocab
		self._doc = doc
		self._options = options

	def to_core(self):
		tokens = self._doc.to_json()["tokens"]

		if self._options.get('ignore_determiners'):
			tokens = [t for t in tokens if t["pos"] != "DET"]

		token_table = TokenTable()
		token_table.extend(self._doc.text, tokens)

		return core.Query(
			self._vocab,
			self._doc.text,
			token_table.to_arrow(),
			**self._options)


class Finder:
	def __init__(self, vocab, corpus):
		self._vocab = vocab
		self._docs = []
		for doc in corpus:
			self._docs.append(
				doc.to_core(len(self._docs), self._vocab)
			)

	def __call__(self, query, n_threads=None, progress=None):
		c_query = query.to_core()

		def find_in_doc(x):
			return x, x.find(c_query)

		total = sum([x.n_tokens for x in self._docs])
		done = 0

		if n_threads is None:
			n_threads = multiprocessing.cpu_count()

		results = None
		with multiprocessing.pool.ThreadPool(processes=n_threads) as pool:
			for doc, r in pool.imap_unordered(find_in_doc, self._docs):
				if results is None:
					results = r
				else:
					results.extend(r)
				done += doc.n_tokens
				if progress:
					progress(done / total)

		return results


class Metric:
	def __init__(self, name):
		pass


class Session:
	def __init__(self, corpus, embeddings):
		self._vocab = core.Vocabulary()
		self._metrics = []
		for embedding in embeddings:
			self._vocab.add_embedding(embedding.to_core())
			self._metrics.append(embedding.as_metric())
		self._finder = Finder(self._vocab, corpus)

	def find(self, doc: spacy.tokens.doc.Doc, metrics=None, options: dict = dict()):
		if metrics is None:
			metrics = self._metrics

		options = options.copy()
		options["metrics"] = metrics

		query = Query(self._vocab, doc, options)
		return self._finder(query)
