import numpy as np
import vectorian.core as core
from cached_property import cached_property
from .token import TokenEmbedding
from ..transform import PCACompression


class ContextualEmbedding(TokenEmbedding):
	def __init__(self, transform=None):
		self._transform = transform

	@property
	def transform(self):
		return self._transform

	@property
	def is_contextual(self):
		return True

	def encode(self, doc):
		raise NotImplementedError()

	def create_instance(self, session):
		# ContextualEmbeddings are their own instance
		return self

	def to_core(self):
		return core.ContextualEmbedding(self.name)

	@property
	def name(self):
		raise NotImplementedError()


class AbstractSpacyEmbedding(ContextualEmbedding):
	def __init__(self, nlp, transform=None):
		super().__init__(transform)
		self._nlp = nlp

	@property
	def nlp(self):
		return self._nlp

	@cached_property
	def name(self):
		meta = self._nlp.meta
		return '/'.join([
			meta['url'], meta['lang'], meta['name'], meta['version']
		] + ([] if self._transform is None else [self._transform.name]))


class SpacyEmbedding(AbstractSpacyEmbedding):
	def __init__(self, nlp, dimension, cache=None, **kwargs):
		super().__init__(nlp, **kwargs)
		self._dimension = dimension
		self._cache = cache

	@property
	def dimension(self):
		return self._dimension

	def pca(self, n_dims):
		return SpacyEmbedding(self._nlp, PCACompression(n_dims))

	def encode(self, doc):
		if self._cache is not None:
			array = self._cache.get(doc.text)
			if array is not None:
				return array

		array = np.array([token.vector for token in self._nlp(doc.text)])

		if self._cache is not None:
			self._cache.put(doc.text, array)

		return array


class SpacyTransformerEmbedding(AbstractSpacyEmbedding):
	@property
	def dimension(self):
		if self._transform is not None:
			return self._transform.dimension
		else:
			# https://spacy.io/usage/processing-pipelines
			# https://thinc.ai/docs/api-model
			tfm = self._nlp.pipeline[self._nlp.pipe_names.index("transformer")][1]
			return tfm.model.get_dim("nO")

	def pca(self, n_dims):
		return SpacyTransformerEmbedding(self._nlp, PCACompression(n_dims))

	def encode(self, doc):
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

