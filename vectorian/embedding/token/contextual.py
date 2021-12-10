import numpy as np
import spacy
import vectorian.core as core
from .token import TokenEmbedding
from ..transform import PCACompression
from ..pipeline import decompose_nlp
from ..encoder import EmbeddingEncoder


class _Impl:
	@property
	def dimension(self):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()

	def encode(self, docs):
		raise NotImplementedError()


class _SpacyImpl(_Impl):
	def __init__(self, nlp):
		self._nlp = nlp

	@property
	def nlp(self):
		return self._nlp


class _VectorImpl(_SpacyImpl):
	def __init__(self, nlp, stats):
		super().__init__(nlp)
		self._stats = stats

	@property
	def dimension(self):
		return self._stats.dimension

	@property
	def name(self):
		return self._stats.name

	def encode(self, docs):
		return np.array([
			[token.vector for token in self._nlp(doc.text)] for doc in docs])


class _TfmImpl(_SpacyImpl):
	@property
	def dimension(self):
		# https://spacy.io/usage/processing-pipelines
		# https://thinc.ai/docs/api-model
		tfm = self._nlp.pipeline[self._nlp.pipe_names.index("transformer")][1]
		return tfm.model.get_dim("nO")

	def _encode(self, doc):
		# https://spacy.io/usage/embeddings-transformers#transformers
		# https://explosion.ai/blog/spacy-transformers
		# https://github.com/explosion/spaCy/issues/6403
		# https://github.com/explosion/spaCy/issues/7032
		# https://github.com/explosion/spaCy/discussions/6511

		if not hasattr(doc._, 'trf_data'):
			raise RuntimeError(
				"Could not access spaCy Transformer data for document. "
				f"Are you sure {self._nlp.meta['name']} is a Transformer model?")

		token_emb, sent_emb = doc._.trf_data.tensors
		token_emb = token_emb.reshape(-1, token_emb.shape[-1])
		n_dims = token_emb.shape[-1]

		trf_vectors = []

		assert len(doc) == len(doc._.trf_data.align)
		for x in doc._.trf_data.align:
			trf_vector = [token_emb[i[0]] for i in x.data]
			if trf_vector:
				trf_vectors.append(np.average(trf_vector, axis=0))
			else:
				trf_vectors.append(np.zeros((n_dims,), dtype=np.float32))

		trf_vectors = np.array(trf_vectors)
		assert len(doc) == trf_vectors.shape[0]

		return trf_vectors

	def encode(self, docs):
		return np.array([self._encode(doc) for doc in docs])


class ContextualEmbeddingEncoder(EmbeddingEncoder):
	def __init__(self, impl, embedding):
		self._impl = impl
		self._embedding = embedding

	@property
	def is_contextual(self):
		return True

	@property
	def name(self):
		return self._embedding.name

	@property
	def embedding(self):
		return self._embedding

	@property
	def dimension(self):
		return self._embedding.dimension

	def encode(self, docs):
		return self._impl.encode(docs)

	def to_core(self):
		return core.ContextualEmbedding(self.name)


class ContextualEmbedding(TokenEmbedding):
	def __init__(self, arg, transform=None):
		if isinstance(arg, _Impl):
			self._impl = arg
		elif isinstance(arg, spacy.Language):
			stats = decompose_nlp(arg)
			if stats is not None:
				self._impl = _VectorImpl(arg, stats)
			elif "transformer" in arg.pipe_names:
				self._impl = _TfmImpl(arg)
			else:
				raise ValueError(
					f"no suitable pipeline component for generating vectors in {arg}")
		else:
			raise TypeError(arg)

		self._transform = transform

	@property
	def transform(self):
		return self._transform

	@property
	def is_contextual(self):
		return True

	def create_encoder(self, session):
		return ContextualEmbeddingEncoder(self._impl, self)

	@property
	def name(self):
		return self._impl.name

	@property
	def dimension(self):
		if self._transform is not None:
			return self._transform.dimension
		else:
			return self._impl.dimension

	def pca(self, n_dims):
		return ContextualEmbedding(
			self._impl, transform=PCACompression(n_dims))
