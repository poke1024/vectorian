import vectorian.core as core
import vectorian.utils as utils
import logging

from cached_property import cached_property
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

	def to_json(self, location_formatter):
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

	@cached_property
	def max_sentence_len(self):
		return max([doc.c_doc.max_sentence_len for doc in self._docs])


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
		for embedding in static_embeddings:
			if not isinstance(embedding, StaticEmbedding):
				raise TypeError(f"expected StaticEmbedding, got {embedding}")
			self._vocab.add_embedding(
				embedding.create_instance(
					self.token_mapper("tokenizer")).to_core())

		self._default_metrics = []
		for embedding in static_embeddings:
			self._default_metrics.append(
				AlignmentSentenceMetric(
					WordSimilarityMetric(
						embedding, CosineMetric())))

		self._collection = Collection(
			self, self._vocab, docs)

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
		for match in self.to_json(self._location_formatter):
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
