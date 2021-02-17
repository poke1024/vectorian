import vectorian.core as core
import spacy
import multiprocessing
import multiprocessing.pool
import logging
import roman

from vectorian.corpus.document import TokenTable
from vectorian.render import Renderer


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


def get_location_desc(metadata, location):
	if location[2] > 0:  # we have an act-scene-speakers structure.
		speaker = metadata["speakers"].get(str(location[2]), "")
		if location[0] >= 0:
			act = roman.toRoman(location[0])
			scene = location[1]
			return speaker, "%s.%d, line %d" % (act, scene, location[3])
		else:
			return speaker, "line %d" % location[3]
	elif location[1] > 0:  # book, chapter and paragraphs
		if location[0] < 0:  # do we have a book?
			return "", "Chapter %d, par. %d" % (location[1], location[3])
		else:
			return "", "Book %d, Chapter %d, par. %d" % (
				location[0], location[1], location[3])
	else:
		return "", "par. %d" % location[3]


def result_set_to_json(result_set):
	matches = []
	for i, m in enumerate(result_set.best_n(-1)):
		regions = []

		try:
			for r in m.regions:
				s = r.s.decode('utf-8', errors='ignore')
				if r.matched:
					t = r.t.decode('utf-8', errors='ignore')
					regions.append(dict(
						s=s,
						t=t,
						similarity=r.similarity,
						weight=r.weight,
						pos_s=r.pos_s.decode('utf-8', errors='ignore'),
						pos_t=r.pos_t.decode('utf-8', errors='ignore'),
						metric=r.metric.decode('utf-8', errors='ignore')))
				else:
					regions.append(dict(s=s, mismatch_penalty=r.mismatch_penalty))

			metadata = m.document.metadata
			speaker, loc_desc = get_location_desc(metadata, m.location)

			matches.append(dict(
				debug=dict(document=m.document.id, sentence=m.sentence_id),
				score=m.score,
				metric=m.metric,
				location=dict(
					speaker=speaker,
					author=metadata["author"],
					title=metadata["title"],
					location=loc_desc
				),
				regions=regions,
				omitted=m.omitted))
		except UnicodeDecodeError:
			logging.exception("unicode conversion issue")

	return matches


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
			n_threads = min(len(self._docs), multiprocessing.cpu_count())

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

		return result_set_to_json(results)


class Session:
	def __init__(self, corpus, embeddings):
		self._vocab = core.Vocabulary()
		self._metrics = []
		for embedding in embeddings:
			self._vocab.add_embedding(embedding.to_core())
			self._metrics.append(embedding.as_metric())
		self._finder = Finder(self._vocab, corpus)

	def find(self, doc: spacy.tokens.doc.Doc, metrics=None, n=100, progress=None, options: dict = dict()):

		if not isinstance(doc, spacy.tokens.doc.Doc):
			raise TypeError("please specify a spaCy document as query")

		if metrics is None:
			metrics = self._metrics

		options = options.copy()
		options["metrics"] = metrics
		options["max_matches"] = n

		query = Query(self._vocab, doc, options)
		return self._finder(query, progress=progress)


class LabResult:
	def __init__(self, data):
		self._data = data

	def _repr_html_(self):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		r = Renderer()
		for result in self._data:
			r.add_match(result)
		return r.to_html()

	def to_json(self):
		return self._data

	def limit_to(self, n):
		return LabResult(self._data[:n])


class LabSession(Session):
	def find(self, *args, return_json=False, **kwargs):
		import ipywidgets as widgets
		from IPython.display import display

		progress = widgets.FloatProgress(
			value=0, min=0, max=1, description="",
			layout=widgets.Layout(width="100%"))

		display(progress)

		def update_progress(t):
			progress.value = progress.max * t

		try:
			results = super().find(
				*args, progress=update_progress, **kwargs)
		finally:
			progress.close()

		return LabResult(results)
