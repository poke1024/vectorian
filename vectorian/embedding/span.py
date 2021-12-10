import numpy as np
import spacy
import cachetools
from vectorian.tqdm import tqdm
from vectorian.corpus import Document
from .pipeline import decompose_nlp
from .vectors import Vectors, ExternalMemoryVectors
from .encoder import EmbeddingEncoder


class _Impl:
	def dimension(self, session):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()

	def encode(self, session, doc_spans):
		raise NotImplementedError()

	@property
	def token_embedding(self):
		raise NotImplementedError()



class AggregatedTokenImpl(_Impl):
	# aggregated token embeddings are used in many publications, e.g.:
	# * Mikolov et al., "Distributed representations of words and
	# phrases and their compositionality.", 2013.
	# * Zhelezniak et al., "DON’T SETTLE FOR AVERAGE, GO FOR THE MAX:
	# FUZZY SETS AND MAX-POOLED WORD VECTORS", 2019.

	_default_functions = {
		np.mean: "mean",
		np.min: "min",
		np.max: "max"
	}

	def __init__(self, embedding, agg=np.mean, agg_name=None):
		super().__init__()

		if agg_name is None:
			agg_name = AggregatedTokenImpl._default_functions.get(agg)
			if agg_name is None:
				raise ValueError(f"cannot obtain automatic name for {agg}")

		self._embedding = embedding
		self._agg = agg
		self._agg_name = agg_name

		if embedding.is_contextual and embedding.transform is not None:
			raise NotImplementedError("cannot use transformed contextual embedding")

	@property
	def name(self):
		return f"aggregated-{self._agg_name}-" + self._embedding.name

	def dimension(self, session):
		return session.to_encoder(self._embedding).dimension

	def encode(self, session, doc_spans):
		encoder = session.to_encoder(self._embedding)

		for doc, spans in doc_spans:
			out = np.empty((len(spans), self.dimension(session)), dtype=np.float32)
			out.fill(np.nan)

			if encoder.is_static:
				for i, span in enumerate(spans):
					text = [token.text for token in span]
					emb_vec = encoder.encode_tokens(text)
					v = emb_vec.unmodified
					if v.shape[0] > 0:
						out[i, :] = self._agg(v, axis=0)

			elif encoder.is_contextual:
				vec_ref = doc.contextual_embeddings[self._embedding.name]
				with vec_ref.open() as emb_vec:
					emb_vec_data = emb_vec.unmodified
					for i, span in enumerate(spans):
						v = emb_vec_data[span.start:span.end, :]
						if v.shape[0] > 0:
							out[i, :] = self._agg(v, axis=0)

			else:
				assert False

			yield out

	@property
	def token_embedding(self):
		return self._embedding


class _PureTextImpl(_Impl):
	def __init__(self, chunk_size=50):
		super().__init__()
		self._chunk_size = chunk_size

	def _encode_text(self, text):
		raise NotImplementedError()

	def dimension(self, session):
		raise NotImplementedError()

	@property
	def token_embedding(self):
		return None  # i.e. no token embedding

	def encode(self, session, doc_spans):
		for doc, spans in doc_spans:
			#for chunk in chunks(spans, self._chunk_size):
			yield self._encode_text([span.text for span in spans])


class _SpacyImpl(_PureTextImpl):
	def __init__(self, nlp, **kwargs):
		super().__init__(**kwargs)
		self._nlp = nlp
		self._stats = decompose_nlp(nlp)
		if self._stats is None:
			raise RuntimeError(f"failed to decompose {nlp.pipeline}")

	def dimension(self, session):
		return self._stats.dimension

	@property
	def name(self):
		return self._stats.name

	def _encode_text(self, texts):
		return [self._nlp(t).vector for t in texts]


class _LambdaImpl(_PureTextImpl):
	def __init__(self, encode, name, vector_size=768, **kwargs):
		super().__init__(**kwargs)
		self._encode = encode
		self._vector_size = vector_size
		self._name = name

	def dimension(self, session):
		return self._vector_size

	def _encode_text(self, texts):
		return self._encode(texts)

	@property
	def name(self):
		return self._name


def chunks(x, n):
	for i in range(0, len(x), n):
		yield x[i:i + n]


def _prepare_doc(doc, nlp):
	if hasattr(doc, 'prepare'):
		if nlp is None:
			raise RuntimeError(f"need nlp to prepare {doc}")
		return doc.prepare(nlp)
	else:
		return doc


def prepare_docs(docs, nlp):
	return [_prepare_doc(doc, nlp) for doc in docs]


class SpanEmbeddingEncoder(EmbeddingEncoder):
	def __init__(self, partition, embedding, impl):
		self._partition = partition
		self._embedding = embedding
		self._impl = impl

	@property
	def name(self):
		return self._impl.name

	@property
	def embedding(self):
		return self._embedding

	@property
	def dimension(self):
		return self._impl.dimension(self._partition.session)

	@property
	def token_embedding(self):
		return self._impl.token_embedding

	def encode(self, docs, pbar=False):
		partition = self._partition

		n_spans = [doc.n_spans(partition) for doc in docs]
		i_spans = np.cumsum([0] + n_spans)

		out = np.empty((i_spans[-1], self.dimension))

		def gen_spans():
			with tqdm(
				desc="Encoding",
				total=i_spans[-1],
				disable=not pbar) as pbar_instance:

				for doc in docs:
					spans = list(doc.spans(partition))
					yield doc, spans
					pbar_instance.update(len(spans))

		for i, v in enumerate(self._impl.encode(partition.session, gen_spans())):
			out[i_spans[i]:i_spans[i + 1], :] = v

		return out


class CachedSpanEmbeddingEncoder(EmbeddingEncoder):
	def __init__(self, partition, embedding, impl, cache_size=150):
		self._partition = partition
		self._corpus = partition.session.corpus
		self._catalog = self._corpus.embedding_catalog
		self._encoder = SpanEmbeddingEncoder(partition, embedding, impl)
		self._cache = cachetools.LRUCache(cache_size)
		self._uuid = self._gen_uuid()

	@property
	def name(self):
		return self._encoder.name

	@property
	def embedding(self):
		return self._encoder.embedding

	@property
	def dimension(self):
		return self._encoder.dimension

	@property
	def token_embedding(self):
		return self._encoder.token_embedding

	def _gen_uuid(self):
		return self._catalog.add_embedding("span", {
			'embedding': self.name,
			'partition': self._partition.cache_key
		})

	def _emb_path(self, doc, mutable=False):
		emb_path = Document.embedding_path(self._corpus.get_doc_path(doc.doc))
		if mutable and not emb_path.exists():
			emb_path.mkdir(exist_ok=True, parents=True)
		return emb_path / self._uuid

	def cache(self, docs, partition, pbar=True):
		if len(docs) > self._cache.maxsize:
			raise RuntimeError("cache too small")
		self.encode(docs, pbar=pbar)

	def _mk_cache_key(self, doc):
		if doc.corpus is None:
			return None
		uid = doc.corpus_id
		if uid is None:
			return None
		return uid

	def _load(self, doc):
		cache_key = self._mk_cache_key(doc)
		if cache_key is None:
			return None
		data = self._cache.get(cache_key)
		if data is None:
			p = self._emb_path(doc)
			if p.with_suffix(".h5").exists():
				data = ExternalMemoryVectors.load(p).unmodified
				self._cache[cache_key] = data
		return data

	def _store(self, doc, v_doc):
		cache_key = self._mk_cache_key(doc)
		if cache_key is not None:
			self._cache[cache_key] = v_doc
			p = self._emb_path(doc, mutable=True)
			Vectors(v_doc).save(p)

	def encode(self, docs, pbar=False):
		partition = self._partition

		n_spans = [doc.n_spans(partition) for doc in docs]
		i_spans = np.cumsum([0] + n_spans)

		out = np.empty((sum(n_spans), self.dimension))

		new = []
		index = []

		# we assume all docs stem from the same corpus. otherwise our caching
		# ids would not be reliable.
		for doc in docs:
			if doc.corpus not in (None, self._corpus):
				raise RuntimeError(f"doc {doc} has corpus {doc.corpus}, expected either None or {self._corpus}")

		for i, doc in enumerate(docs):
			cached = self._load(doc)
			if cached is not None:
				out[i_spans[i]:i_spans[i + 1], :] = cached
			else:
				new.append(doc)
				index.append(i)

		if new:
			v = self._encoder.encode(new, pbar)

			n_spans_new = [n_spans[i] for i in index]
			i_spans_new = np.cumsum([0] + n_spans_new)

			for j, i in enumerate(index):
				v_doc = v[i_spans_new[j]:i_spans_new[j + 1], :]
				out[i_spans[i]:i_spans[i + 1], :] = v_doc
				self._store(docs[i], v_doc)

		return out


class SpanEmbedding:
	def __init__(self, arg, cached=True):
		if isinstance(arg, _Impl):
			self._impl = arg
		elif isinstance(arg, spacy.Language):
			self._impl = _SpacyImpl(arg)
		else:
			raise TypeError(arg)
		self._cached = cached

	def create_encoder(self, partition, ):
		if self._cached:
			klass = CachedSpanEmbeddingEncoder
		else:
			klass = SpanEmbeddingEncoder
		return klass(
			partition, self, self._impl)

	def to_sentence_sim(self, vector_sim=None):
		from vectorian.sim.span import EmbeddedSpanSim
		return EmbeddedSpanSim(self, vector_sim)


class SentenceEmbedding(SpanEmbedding):
	pass
