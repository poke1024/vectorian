import vectorian.core as core
import logging
import roman
import re

from cached_property import cached_property
from functools import lru_cache

from vectorian.render import Renderer, LocationFormatter
from vectorian.metrics import CosineMetric, WordSimilarityMetric, AlignmentSentenceMetric, SentenceSimilarityMetric
from vectorian.embeddings import StaticEmbedding


class Result:
	def __init__(self, index, matches, duration):
		self._index = index
		self._matches = matches
		self._duration = duration

	@property
	def index(self):
		return self._index

	def __iter__(self):
		return self._matches

	def __getitem__(self, i):
		return self._matches[i]

	@lru_cache(1)
	def to_json(self):
		return [m.to_json(self._index.session) for m in self._matches]

	def limit_to(self, n):
		return type(self)(self._matches[:n])

	@property
	def duration(self):
		return self._duration


class CompiledDoc:
	def __init__(self, p_doc, args):
		self._p_doc = p_doc
		self._args = args

	@property
	def p_doc(self):
		return self._p_doc

	@cached_property
	def c_doc(self):
		return self._p_doc.to_core(*self._args)


class Collection:
	def __init__(self, vocab, corpus, token_filter):
		self._vocab = vocab
		self._docs = []
		for doc in corpus:
			p_doc = doc.prepare(token_filter)
			self._docs.append(CompiledDoc(
				p_doc, (len(self._docs), self._vocab)))

	@property
	def documents(self):
		return self._docs

	@cached_property
	def max_sentence_len(self):
		return max([doc.c_doc.max_sentence_len for doc in self._docs])


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
		# which really breaks pos_mismatch_penalty often. we re-
		# classify PROPN as NOUN.
		t_new = t.copy()
		t_new["pos"] = self._pos.get(t["pos"], t["pos"])
		t_new["tag"] = self._tag.get(t["tag"], t["tag"])
		return t_new


class TokenNormalizer:
	def __init__(self):
		self._pattern = re.compile(r"[^\w]")

	def __call__(self, token):
		return self._pattern.sub("", token.lower())


class Session:
	def __init__(
			self, docs, static_embeddings=[],
			import_filter=None, location_formatter=None):

		if import_filter is None:
			import_filter = DefaultImportFilter()
		if location_formatter is None:
			location_formatter = LocationFormatter()

		self._vocab = core.Vocabulary()
		self._default_metrics = []
		for embedding in static_embeddings:
			if not isinstance(embedding, StaticEmbedding):
				raise TypeError(f"expected StaticEmbedding, got {embedding}")
			self._vocab.add_embedding(embedding.to_core())
			self._default_metrics.append(
				AlignmentSentenceMetric(
					WordSimilarityMetric(
						embedding, CosineMetric())))
		self._collection = Collection(self._vocab, docs, import_filter)
		self._location_formatter = location_formatter

	@cached_property
	def documents(self):
		return [x.p_doc for x in self._collection.documents]

	@cached_property
	def c_documents(self):
		return [x.c_doc for x in self._collection.documents]

	@property
	def vocab(self):
		return self._vocab

	@property
	def max_sentence_len(self):
		return self._collection.max_sentence_len

	@property
	def result_class(self):
		return Result

	def index_for_metric(self, metric=None):
		if metric is None:
			metric = self._default_metrics[0]
		assert isinstance(metric, SentenceSimilarityMetric)
		return metric.create_index(self)

	def run_query(self, find, query):
		return Result, find(query)

	@property
	def location_formatter(self):
		return self._location_formatter


class LabResult(Result):
	def __init__(self, index, matches, duration, annotate=None):
		super().__init__(index, matches, duration)
		self._annotate = annotate

	def _render(self, r):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		for match in self.to_json():
			r.add_match(match)
		return r.to_html()

	def annotate(self, tags=True, metric=True, penalties=True, **kwargs):
		return LabResult(
			self.index,
			self._matches,
			self._duration,
			annotate=dict(tags=tags, metric=metric, penalties=penalties, **kwargs))

	def _repr_html_(self):
		return self._render(Renderer(annotate=self._annotate))


class LabSession(Session):
	def run_query(self, find, query):
		import ipywidgets as widgets
		from IPython.display import display

		progress = widgets.FloatProgress(
			value=0, min=0, max=1, description="",
			layout=widgets.Layout(width="100%"))

		display(progress)

		def update_progress(t):
			progress.value = progress.max * t

		try:
			result = find(
				query,
				progress=update_progress)
		finally:
			progress.close()

		return LabResult, result
