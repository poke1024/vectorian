import vectorian.core as core
import vectorian.utils as utils
import logging

from cached_property import cached_property
from functools import lru_cache
from vectorian.render.render import Renderer
from vectorian.render.excerpt import ExcerptRenderer
from vectorian.render.location import LocationFormatter
from vectorian.metrics import CosineSimilarity, TokenSimilarity, AlignmentSentenceSimilarity, SentenceSimilarity
from vectorian.embeddings import StaticEmbedding, VectorsCache


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

	def __len__(self):
		return len(self._matches)

	def __iter__(self):
		return self._matches

	def __getitem__(self, i):
		return self._matches[i]

	def to_json(self, context_size=10):
		return [m.to_json(context_size) for m in self._matches]

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
		if not isinstance(metric, SentenceSimilarity):
			raise TypeError(metric)

		if nlp:
			kwargs = kwargs.copy()
			kwargs['nlp'] = nlp

		return metric.create_index(self, **kwargs)


class Session:
	def __init__(self, docs, embeddings=None, token_mappings=None):
		if embeddings is None:
			embeddings = []

		if token_mappings == "default":
			token_mappings = utils.default_token_mappings()

		self._vocab = core.Vocabulary()

		if any(e.is_static for e in embeddings) and not token_mappings:
			logging.warning("got static embeddings but not token mappings.")

		if token_mappings is None:
			token_mappings = {}
		self._token_mappings = token_mappings

		if any(k not in ("tokenizer", "tagger") for k in token_mappings):
			raise ValueError(token_mappings)

		self._embeddings = {}
		for embedding in embeddings:
			self._embeddings[embedding.name] = embedding

		self._embedding_instances = {}
		for embedding in embeddings:
			if embedding.is_static:
				instance = embedding.create_instance(self)
				self._embedding_instances[instance.name] = instance
				self._vocab.add_embedding(instance)

		for embedding in embeddings:
			if embedding.is_contextual:
				for doc in docs:
					if not doc.has_contextual_embedding(embedding.name):
						raise RuntimeError(f"doc {doc.unique_id} misses contextual embedding {embedding.name}")

		self._collection = Collection(
			self, self._vocab, docs)

		# make sure all core.Documents are instantiated at this point so that
		# the interval vocabulary is setup and complete.
		self.c_documents

		self._vocab.compile_embeddings()  # i.e. static embeddings

		self._vectors_cache = VectorsCache()

	def default_metric(self):
		embedding = list(self._embeddings.values())[0]
		return AlignmentSentenceSimilarity(
			TokenSimilarity(
				embedding, CosineSimilarity()))

	@cached_property
	def documents(self):
		return [x.p_doc for x in self._collection.documents]

	@cached_property
	def c_documents(self):
		return [x.c_doc for x in self._collection.documents]

	@property
	def embeddings(self):
		return list(self._embeddings.values())

	def get_embedding_instance(self, embedding):
		return self._embedding_instances[embedding.name]

	@property
	def vocab(self):
		return self._vocab

	@property
	def vectors_cache(self):
		return self._vectors_cache

	@lru_cache(16)
	def max_len(self, level, window_size):
		return self._collection.max_len(level, window_size)

	def make_result(self, *args, **kwargs):
		return Result(*args, **kwargs)

	def on_progress(self, task):
		return task(None)

	def partition(self, level, window_size=1, window_step=None):
		if window_step is None:
			window_step = window_size
		return Partition(self, level, window_size, window_step)

	def token_mapper(self, stage):
		if stage not in ('tokenizer', 'tagger'):
			raise ValueError(stage)
		if stage == 'tokenizer':
			return utils.CachableCallable.chain(
				self._token_mappings.get('tokenizer', []))
		else:
			return utils.chain(self._token_mappings.get(stage, []))


class LabResult(Result):
	def __init__(self, index, matches, duration, renderers, location_formatter):
		super().__init__(index, matches, duration)
		self._renderers = renderers
		self._location_formatter = location_formatter
		self._annotate = {}

	def _render(self, r):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		return r.to_html(self._matches)

	def format(self, render_spec):
		renderers = []
		if isinstance(render_spec, (list, tuple)):
			renderers = render_spec
		else:
			def load_excerpt_renderer():
				from vectorian.render.excerpt import ExcerptRenderer
				return ExcerptRenderer

			def load_flow_renderer():
				from vectorian.render.sankey import FlowRenderer
				return FlowRenderer

			def load_matrix_renderer():
				from vectorian.render.matrix import MatrixRenderer
				return MatrixRenderer

			lookup = {
				'excerpt': load_excerpt_renderer,
				'flow': load_flow_renderer,
				'matrix': load_matrix_renderer
			}

			klass = None
			args = []
			for render_desc in render_spec.split(","):
				for i, part in enumerate(render_desc.split()):
					part = part.strip()
					if i == 0:
						klass = lookup[part]()
						args = []
					else:
						if part.startswith("+"):
							args.append(part[1:].strip())
						else:
							raise ValueError(part)

				if klass is not None:
					renderers.append(klass(*args))
					klass = None

			if klass is not None:
				renderers.append(klass(*args))
				klass = None

		return LabResult(
			self.index,
			self._matches,
			self._duration,
			renderers,
			self._location_formatter)

	def _repr_html_(self):
		return self._render(Renderer(
			self._renderers,
			self._location_formatter,
			annotate=self._annotate))


class LabSession(Session):
	def __init__(self, *args, location_formatter=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._location_formatter = location_formatter or LocationFormatter()

	def interact(self):
		pass  # return InteractiveQuery(self)

	def make_result(self, *args, **kwargs):
		return LabResult(
			*args, **kwargs,
			renderers=[ExcerptRenderer()],
			location_formatter=self._location_formatter)

	def on_progress(self, task):
		import ipywidgets as widgets
		from IPython.display import display

		progress = widgets.FloatProgress(
			value=0, min=0, max=1, description="",
			layout=widgets.Layout(width="100%"))

		display(progress)

		def update_progress(t):
			progress.value = progress.max * t

		try:
			result = task(update_progress)
		finally:
			progress.close()

		return result
