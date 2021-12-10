import vectorian.core as core
import logging
import collections
import time
import numpy as np
import concurrent.futures

from cached_property import cached_property
from functools import lru_cache

from vectorian.render.render import Renderer
from vectorian.render.excerpt import ExcerptRenderer
from vectorian.render.location import LocationFormatter
from vectorian.sim.vector import CosineSim
from vectorian.sim.token import EmbeddingTokenSim
from vectorian.sim.span import SpanSim, OptimizedSpanSim
from vectorian.embedding.vectors import OpenedVectorsCache, Vectors
from vectorian.embedding.token import TokenEmbedding
from vectorian.normalization import Normalization, VanillaNormalization
from vectorian.corpus.corpus import Corpus
from vectorian.tqdm import tqdm


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


class Collection:
	def __init__(self, session, vocab, corpus):
		self._vocab = vocab
		self._docs = []

		with corpus.flavor_cache(session.normalization.name) as flavor_cache:

			with tqdm(desc="Preparing Documents", total=len(corpus)) as pbar:
				def prepare_doc(doc):
					pbar.update(1)
					return doc.prepare(corpus, flavor_cache, session)

				with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
					self._docs = list(executor.map(prepare_doc, corpus.docs))

	@property
	def documents(self):
		return self._docs

	@lru_cache(16)
	def max_len(self, level, window_size):
		return max([doc.compiled.max_len(level, window_size) for doc in self._docs])


Slice = collections.namedtuple('Slice', ['level', 'start', 'end'])


class Partition:
	def __init__(self, session, level, window_size, window_step):
		self._session = session
		self._level = level
		self._window_size = window_size
		self._window_step = window_step

	@property
	def contiguous(self):
		return self._window_step <= self._window_size

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

	@property
	def cache_key(self):
		return self._level, self._window_size, self._window_step

	@cached_property
	def freq(self):
		freq = core.Frequencies(self._session.vocab)
		strategy = core.SliceStrategy(self.to_args())
		for doc in self._session.documents:
			freq.add(doc.compiled, strategy)
		return freq

	def max_len(self):
		return self._session.max_len(self._level, self._window_size)

	def index(self, metric, nlp=None, **kwargs):
		if not isinstance(metric, SpanSim):
			raise TypeError(metric)

		if nlp:
			kwargs = kwargs.copy()
			kwargs['nlp'] = nlp

		return metric.create_index(self, **kwargs)

	def slice_id_to_slice(self, slice_id):
		return Slice(self._level, self._window_step * slice_id, self._window_size)


def _default_normalization(corpus: Corpus, normalization:Normalization = None):
	if normalization is None:
		c_norms = corpus.normalizations
		if not c_norms or tuple(c_norms) == ("vanilla",):
			normalization = VanillaNormalization()
			if not c_norms:
				corpus.add_normalization(normalization)
			return normalization
		else:
			raise ValueError("please specify a Normalization")

	if normalization.name not in corpus.flavors:
		raise ValueError(f"Normalization '{normalization.name}' does not exist in corpus")

	return normalization


class Session:
	def __init__(self, corpus: Corpus, embeddings=None, normalization:Normalization = None):
		self._corpus = corpus
		self._normalization = _default_normalization(corpus, normalization)

		if embeddings is None:
			embeddings = []

		self._embedding_manager = core.EmbeddingManager()

		self._token_embeddings = tuple(filter(lambda x: isinstance(x, TokenEmbedding), embeddings))

		for embedding in self._token_embeddings:
			if embedding.is_contextual:
				for doc in corpus:
					if not doc.has_contextual_embedding(embedding.name):
						unique_id = corpus.get_unique_id(doc)
						raise RuntimeError(f"doc {unique_id} misses contextual embedding {embedding.name}")

		self._embedding_encoders = collections.OrderedDict()
		for embedding in self._token_embeddings:
			encoder = embedding.create_encoder(self)
			self._embedding_encoders[encoder.name] = encoder
			self._embedding_manager.add_embedding(encoder)

		self._vocab = core.Vocabulary(self._embedding_manager)

		self._collection = Collection(
			self, self._vocab, corpus)

		self._vocab.compile_embeddings()  # i.e. static token embeddings
		self._embedding_manager.compile_contextual()

		self._vectors_cache = OpenedVectorsCache()

	@property
	def corpus(self):
		return self._corpus

	@property
	def vocab(self):
		return self._vocab

	@property
	def normalization(self):
		return self._normalization

	@property
	def normalizers(self):
		return self._normalization.normalizers

	def default_metric(self):
		embedding = self._token_embeddings[0]
		return OptimizedSpanSim(
			EmbeddingTokenSim(
				embedding, CosineSim()))

	@cached_property
	def documents(self):
		return self._collection.documents

	@cached_property
	def c_documents(self):
		return [x.compiled for x in self.documents]

	@property
	def encoders(self):
		return self._embedding_encoders

	def to_encoder(self, embedding):
		return self._embedding_encoders[embedding.name]

	def cache_contextual_embeddings(self):
		for doc in tqdm(self.documents, desc="Loading Vectors"):
			doc.cache_contextual_embeddings()

	@property
	def vectors_cache(self):
		return self._vectors_cache

	@lru_cache(16)
	def max_len(self, level, window_size):
		return self._collection.max_len(level, window_size)

	def make_result(self, *args, **kwargs):
		return Result(*args, **kwargs)

	def on_progress(self, task, disable_progress=False):
		return task(None)

	def partition(self, level, window_size=1, window_step=None):
		if window_step is None:
			window_step = window_size
		return Partition(self, level, window_size, window_step)

	def index(self, *args, **kwargs):
		return self.partition("sentence").index(*args, **kwargs)

	def word_vec(self, embedding, token_or_tokens):
		from vectorian.corpus.document import Token

		if embedding.is_static:
			encoder = self.to_encoder(embedding)
			def get(token):
				if isinstance(token, Token):
					token = token.text
				return encoder.word_vec(token)
		elif embedding.is_contextual:
			def get(token):
				if not isinstance(token, Token):
					raise ValueError(f"expected a Token, got {token}")

				with token.doc.contextual_embeddings[embedding.name].open() as vec:
					return vec.unmodified[token.index]
		else:
			raise ValueError()

		if isinstance(token_or_tokens, (list, tuple)):
			return [get(x) for x in token_or_tokens]
		else:
			return get(token_or_tokens)

	def similarity(self, token_sim, a, b):
		from vectorian.corpus.document import Token

		out = np.zeros((1, 1), dtype=np.float32)
		if token_sim.is_modifier:
			x = np.zeros((len(token_sim.operands), 1), dtype=np.float32)
			for i, op in enumerate(token_sim.operands):
				x[i] = self.similarity(op, a, b)
			token_sim(x, out)

		else:
			encoder = self.to_encoder(
				token_sim.embedding)
			if encoder.is_static:
				if isinstance(a, Token):
					a = a.text
				if isinstance(b, Token):
					b = b.text

				va = Vectors([encoder.word_vec(a)])
				vb = Vectors([encoder.word_vec(b)])

			elif encoder.is_contextual:
				if not isinstance(a, Token):
					raise ValueError(f"expected a Token, got {a}")
				if not isinstance(b, Token):
					raise ValueError(f"expected a Token, got {b}")

				with a.doc.contextual_embeddings[encoder.name].open() as vec:
					va = Vectors([vec.unmodified[a.index]])

				with b.doc.contextual_embeddings[encoder.name].open() as vec:
					vb = Vectors([vec.unmodified[b.index]])
			else:
				raise ValueError()

			token_sim.similarity(va, vb, out)

		return out[0, 0]


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
		self._progress = None
		self._last_progress_update = None

	def interact(self, nlp, **kwargs):
		from vectorian.interact import InteractiveQuery

		logger = logging.getLogger()
		logger.setLevel(logging.WARNING)

		q = InteractiveQuery(self, nlp, **kwargs)
		return q.widget

	def make_result(self, *args, **kwargs):
		return LabResult(
			*args, **kwargs,
			renderers=[ExcerptRenderer()],
			location_formatter=self._location_formatter)

	def _create_progress(self):
		import ipywidgets as widgets
		from IPython.display import display

		if self._progress is not None:
			return

		self._progress = widgets.FloatProgress(
			value=0, min=0, max=1, description="",
			layout=widgets.Layout(width="100%"))

		display(self._progress)

	def _update_progress(self, t):
		update_delay = 0.5

		now = time.time()
		if now - self._last_progress_update < update_delay:
			return

		self._create_progress()

		new_value = self._progress.max * t
		self._progress.value = new_value
		self._last_progress_update = now

	def on_progress(self, task, disable_progress=False):
		self._last_progress_update = time.time()

		try:
			if disable_progress:
				result = task(None)
			else:
				result = task(self._update_progress)
		finally:
			if self._progress:
				self._progress.close()
				self._progress = None

		return result

