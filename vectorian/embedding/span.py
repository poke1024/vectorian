import numpy as np


class SpanEmbedding:
	def __init__(self):
		pass

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def embedding(self):  # i.e. token embedding
		raise NotImplementedError()

	def encode(self, session, doc_spans):
		raise NotImplementedError()

	def to_sentence_sim(self, vector_sim=None):
		from vectorian.sim.span import SpanSimFromSpanEmbeddings

		return SpanSimFromSpanEmbeddings(self, vector_sim)


class AggregatedTokenEmbedding(SpanEmbedding):
	# aggregated token embeddings are used in many publications, e.g.:
	# * Mikolov et al., "Distributed representations of words and
	# phrases and their compositionality.", 2013.
	# * Zhelezniak et al., "DON’T SETTLE FOR AVERAGE, GO FOR THE MAX:
	# FUZZY SETS AND MAX-POOLED WORD VECTORS", 2019.

	def __init__(self, embedding, agg=np.mean):
		super().__init__()
		self._embedding = embedding
		self._agg = agg

		if embedding.is_contextual and embedding.transform is not None:
			raise NotImplementedError("cannot use transformed contextual embedding")

	@property
	def embedding(self):
		return self._embedding

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


class FullTextSpanEmbedding(SpanEmbedding):
	def __init__(self, chunk_size=50):
		super().__init__()
		self._chunk_size = chunk_size

	def _encode_text(self, text):
		raise NotImplementedError()

	def vector_size(self, session):
		raise NotImplementedError()

	@property
	def embedding(self):
		return None  # i.e. no token embedding

	def encode(self, session, doc_spans):
		for doc, spans in doc_spans:
			#for chunk in chunks(spans, self._chunk_size):
			yield self._encode_text([span.text for span in spans])


class TextEncoderEmbedding(FullTextSpanEmbedding):
	def __init__(self, encode, vector_size=768, **kwargs):
		super().__init__(**kwargs)
		self._encode = encode
		self._vector_size = vector_size

	def vector_size(self, session):
		return self._vector_size

	def _encode_text(self, texts):
		return self._encode(texts)
