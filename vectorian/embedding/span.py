import numpy as np
from .pipeline import decompose_nlp


class SpanEmbedding:
	def __init__(self):
		pass

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def token_embedding(self):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()

	def encode(self, session, doc_spans):
		raise NotImplementedError()

	def to_sentence_sim(self, vector_sim=None):
		from vectorian.sim.span import EmbeddedSpanSim
		return EmbeddedSpanSim(self, vector_sim)


class AggregatedTokenEmbedding(SpanEmbedding):
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
			agg_name = AggregatedTokenEmbedding._default_functions.get(agg)
			if agg_name is None:
				raise ValueError(f"cannot obtain automatic name for {agg}")

		self._embedding = embedding
		self._agg = agg
		self._agg_name = agg_name

		if embedding.is_contextual and embedding.transform is not None:
			raise NotImplementedError("cannot use transformed contextual embedding")

	@property
	def token_embedding(self):
		return self._embedding

	@property
	def name(self):
		return f"aggregated-{self._agg_name}-" + self._embedding.name

	def vector_size(self, session):
		return session.to_embedding_instance(self._embedding).dimension

	def encode(self, session, doc_spans):
		embedding = session.to_embedding_instance(self._embedding)

		for doc, spans in doc_spans:
			out = np.empty((len(spans), self.vector_size(session)), dtype=np.float32)
			out.fill(np.nan)

			if embedding.is_static:
				for i, span in enumerate(spans):
					text = [token.text for token in span]
					emb_vec = embedding.get_embeddings(text)
					v = emb_vec.unmodified
					if v.shape[0] > 0:
						out[i, :] = self._agg(v, axis=0)

			elif embedding.is_contextual:
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


class PureSpanEmbedding(SpanEmbedding):
	def __init__(self, chunk_size=50):
		super().__init__()
		self._chunk_size = chunk_size

	def _encode_text(self, text):
		raise NotImplementedError()

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def token_embedding(self):
		return None  # i.e. no token embedding

	def encode(self, session, doc_spans):
		for doc, spans in doc_spans:
			#for chunk in chunks(spans, self._chunk_size):
			yield self._encode_text([span.text for span in spans])


class SpacySpanEmbedding(PureSpanEmbedding):
	def __init__(self, nlp, **kwargs):
		super().__init__(**kwargs)
		self._nlp = nlp
		self._stats = decompose_nlp(nlp)

	def vector_size(self, session):
		return self._stats.dimension

	@property
	def name(self):
		return self._stats.name

	def _encode_text(self, texts):
		return [self._nlp(t).vector for t in texts]


class LambdaSpanEmbedding(PureSpanEmbedding):
	def __init__(self, encode, name, vector_size=768, **kwargs):
		super().__init__(**kwargs)
		self._encode = encode
		self._vector_size = vector_size
		self._name = name

	def vector_size(self, session):
		return self._vector_size

	def _encode_text(self, texts):
		return self._encode(texts)

	@property
	def name(self):
		return self._name
