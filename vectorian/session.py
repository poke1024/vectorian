import vectorian.core as core
import vectorian.utils as utils
import logging

from cached_property import cached_property
from functools import lru_cache
from pathlib import Path
from vectorian.render import Renderer, LocationFormatter
from vectorian.metrics import CosineMetric, TokenSimilarityMetric, AlignmentSentenceMetric, SentenceSimilarityMetric
from vectorian.embeddings import StaticEmbedding


class Result:
	def __init__(self, index, matches, duration):
		self._index = index
		self._matches = matches
		self._duration = duration

	@property
	def index(self):
		return self._index

	@property
	def matches(self):
		return self._matches

	def __iter__(self):
		return self._matches

	def __getitem__(self, i):
		return self._matches[i]

	def to_json(self, location_formatter=None):
		return [m.to_json(location_formatter) for m in self._matches]

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
	def __init__(self, session, vocab, corpus):
		self._vocab = vocab
		self._docs = []
		for doc in corpus:
			p_doc = doc.prepare(session)
			self._docs.append(CompiledDoc(
				p_doc, (len(self._docs), self._vocab)))

	@property
	def documents(self):
		return self._docs

	@lru_cache(16)
	def max_len(self, level, window_size):
		return max([doc.c_doc.max_len(level, window_size) for doc in self._docs])


class Partition:
	def __init__(self, session, level, window_size, window_step):
		self._session = session
		self._level = level
		self._window_size = window_size
		self._window_step = window_step

	@property
	def session(self):
		return self._session

	@property
	def level(self):
		return self._level

	@property
	def window_size(self):
		return self._window_size

	@property
	def window_step(self):
		return self._window_step

	def to_args(self):
		return {
			'level': self._level,
			'window_size': self._window_size,
			'window_step': self._window_step
		}

	def max_len(self):
		return self._session.max_len(self._level, self._window_size)

	def index(self, metric, nlp=None, **kwargs):
		if not isinstance(metric, SentenceSimilarityMetric):
			raise TypeError(metric)

		if nlp:
			kwargs = kwargs.copy()
			kwargs['nlp'] = nlp

		return metric.create_index(self, **kwargs)


class Session:
	def __init__(self, docs, static_embeddings=None, token_mappings=None):
		self._vocab = core.Vocabulary()

		if static_embeddings and not token_mappings:
			logging.warn("got static embeddings but not token mappings.")

		if token_mappings is None:
			token_mappings = {}
		self._token_mappings = token_mappings

		if any(k not in ("tokenizer", "tagger") for k in token_mappings):
			raise ValueError(token_mappings)

		if static_embeddings is None:
			static_embeddings = []
		self._embeddings = {}
		for embedding in static_embeddings:
			if not isinstance(embedding, StaticEmbedding):
				raise TypeError(f"expected StaticEmbedding, got {embedding}")
			instance = embedding.create_instance(
				self.token_mapper("tokenizer"))
			self._embeddings[instance.name] = instance
			self._vocab.add_embedding(instance.to_core())

		self._collection = Collection(
			self, self._vocab, docs)

	def default_metric(self):
		embedding = list(self._embeddings.values())[0]
		return AlignmentSentenceMetric(
			TokenSimilarityMetric(
				embedding, CosineMetric()))

	@cached_property
	def documents(self):
		return [x.p_doc for x in self._collection.documents]

	@cached_property
	def c_documents(self):
		return [x.c_doc for x in self._collection.documents]

	def get_embedding_instance(self, embedding):
		return self._embeddings[embedding.name]

	@property
	def vocab(self):
		return self._vocab

	@lru_cache(16)
	def max_len(self, level, window_size):
		return self._collection.max_len(level, window_size)

	@property
	def result_class(self):
		return Result

	def partition(self, level, window_size=1, window_step=None):
		if window_step is None:
			window_step = window_size
		return Partition(self, level, window_size, window_step)

	def run_query(self, find, query):
		return Result, find(query)

	def token_mapper(self, stage):
		if stage not in ('tokenizer', 'tagger'):
			raise ValueError(stage)
		if stage == 'tokenizer':
			return utils.CachableCallable.chain(
				self._token_mappings.get('tokenizer', []))
		else:
			return utils.chain(self._token_mappings.get(stage, []))


class LabResult(Result):
	def __init__(self, index, matches, duration, location_formatter, annotate=None):
		super().__init__(index, matches, duration)
		self._annotate = annotate
		self._location_formatter = location_formatter

	def _render(self, r):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		return r.to_html(self._matches)

	def annotate(self, *args):
		# e.g. tags, metric, penalties, metadata, flow
		return LabResult(
			self.index,
			self._matches,
			self._duration,
			self._location_formatter,
			annotate=dict((k, True) for k in args))

	def _repr_html_(self):
		return self._render(Renderer(self._location_formatter, annotate=self._annotate))


class LabSession(Session):
	def __init__(self, *args, location_formatter=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._location_formatter = location_formatter or LocationFormatter()

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

		def make_result(*args, **kwargs):
			return LabResult(*args, **kwargs, location_formatter=self._location_formatter)

		return make_result, result
