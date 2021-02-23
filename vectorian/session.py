import vectorian.core as core
import spacy
import multiprocessing
import multiprocessing.pool
import logging
import roman
import time

from cached_property import cached_property
from functools import lru_cache

from vectorian.corpus.document import TokenTable
from vectorian.render import Renderer
from vectorian.alignment import WatermanSmithBeyer
from vectorian.metrics import CosineMetric, IsolatedMetric, SentenceMetric


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


def get_location_desc(metadata, location):
	book = location["book"]
	chapter = location["chapter"]
	speaker = location["speaker"]
	paragraph = location["paragraph"]

	if speaker > 0:  # we have an act-scene-speakers structure.
		speaker = metadata["speakers"].get(str(speaker), "")
		if book >= 0:
			act = roman.toRoman(book)
			scene = chapter
			return speaker, "%s.%d, line %d" % (act, scene, paragraph)
		else:
			return speaker, "line %d" % paragraph
	elif chapter > 0:  # book, chapter and paragraphs
		if book < 0:  # do we have a book?
			return "", "Chapter %d, par. %d" % (chapter, paragraph)
		else:
			return "", "Book %d, Chapter %d, par. %d" % (
				book, chapter, paragraph)
	else:
		return "", "par. %d" % paragraph


def result_set_to_json(items):
	matches = []
	for i, m in enumerate(items):
		regions = []
		doc = m.document
		sentence = doc.sentence(m.sentence)

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
			speaker, loc_desc = get_location_desc(metadata, sentence)

			matches.append(dict(
				debug=dict(document=m.document.id, sentence=m.sentence),
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


class Result:
	def __init__(self, matches, duration):
		self._matches = matches
		self._duration = duration

	def __iter__(self):
		return self._matches

	def __getitem__(self, i):
		return self._matches[i]

	@lru_cache(1)
	def to_json(self):
		return result_set_to_json(self._matches)

	def limit_to(self, n):
		return type(self)(self._matches[:n])

	@property
	def duration(self):
		return self._duration


class Finder:
	def __init__(self, vocab, corpus, filter_):
		self._vocab = vocab
		self._docs = []
		for doc in corpus:
			self._docs.append(
				doc.to_core(len(self._docs), self._vocab, filter_)
			)

	@property
	def documents(self):
		return self._docs

	@cached_property
	def max_sentence_len(self):
		return max([doc.max_sentence_len for doc in self._docs])

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

		return results


class DefaultImportFilter:
	_pos = {
		'PROPN': 'NOUN'
	}

	_tag = {
		'NNP': 'NN',
		'NNPS': 'NNS',
	}

	def __call__(self, t):
		if t["pos"] == "PUNCT":
			return False

		# spaCy is very generous with labeling things as PROPN,
		# which really breaks pos_mimatch_penalty often. we re-
		# classify PROPN as NOUN.
		t_new = t.copy()
		t_new["pos"] = self._pos.get(t["pos"], t["pos"])
		t_new["tag"] = self._tag.get(t["tag"], t["tag"])
		return t_new


class Session:
	def __init__(self, corpus, embeddings, import_filter=DefaultImportFilter()):
		self._vocab = core.Vocabulary()
		self._default_metrics = []
		for embedding in embeddings:
			self._vocab.add_embedding(embedding.to_core())
			self._default_metrics.append(
				IsolatedMetric(CosineMetric(embedding)))
		self._finder = Finder(self._vocab, corpus, import_filter)

	@property
	def documents(self):
		return self._finder.documents

	@property
	def max_sentence_len(self):
		return self._finder.max_sentence_len

	def find(
		self, doc: spacy.tokens.doc.Doc, alignment=None, metric=None,
		n=100, min_score=0.2, progress=None, ret_class=Result,
		options: dict = dict()):

		if not isinstance(doc, spacy.tokens.doc.Doc):
			raise TypeError("please specify a spaCy document as query")

		if alignment is None:
			alignment = WatermanSmithBeyer()

		metrics = options.get("metrics")
		if metrics is None and metric is not None:
			metrics = [metric]
		if metrics is None:
			metrics = self._default_metrics
		assert all(isinstance(x, SentenceMetric) for x in metrics)

		options = options.copy()
		options["metrics"] = [m.to_args() for m in metrics]
		options["alignment"] = alignment.to_args(self)
		options["max_matches"] = n
		options["min_score"] = min_score

		start_time = time.time()

		query = Query(self._vocab, doc, options)
		r = self._finder(query, progress=progress)

		return ret_class(
			r.best_n(-1),
			duration=time.time() - start_time)


class LabResult(Result):
	def __init__(self, matches, duration, annotate=None):
		super().__init__(matches, duration)
		self._annotate = annotate

	def _render(self, r):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		for match in self.to_json():
			r.add_match(match)
		return r.to_html()

	def annotate(self, tags=True, metric=True, **kwargs):
		return LabResult(
			self._matches,
			self._duration,
			annotate=dict(tags=tags, metric=metric, **kwargs))

	def _repr_html_(self):
		return self._render(Renderer(annotate=self._annotate))


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
			result = super().find(
				*args,
				progress=update_progress,
				ret_class=LabResult,
				**kwargs)
		finally:
			progress.close()

		return result
