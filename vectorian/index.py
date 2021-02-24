import vectorian.core as core
import spacy
import multiprocessing
import multiprocessing.pool
import time

from vectorian.corpus.document import TokenTable


class Query:
	def __init__(self, vocab, doc, options):
		self._vocab = vocab
		self._doc = doc
		self._options = options

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

		options = options.copy()
		options["metric"] = self._metric.to_args(self._session)
		options["max_matches"] = n
		options["min_score"] = min_score

		start_time = time.time()

		query = Query(self._session.vocab, doc, options)
		result_class, r = self._session.run_query(self._find, query)

		return result_class(
			r.best_n(-1),
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

		return results
