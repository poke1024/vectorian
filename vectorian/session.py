import vectorian.core as core
import spacy

from vectorian.query import Query
from vectorian.corpus.corpus import Corpus
from vectorian.corpus.document import Document


class Session:
	def __init__(self, documents, embeddings):
		self._vocab = core.Vocabulary()
		self._metrics = []
		for embedding in embeddings:
			self._vocab.add_embedding(embedding.to_core())
			self._metrics.append(embedding.name)
		self._corpus = Corpus(self._vocab)
		for doc in documents:
			self._corpus.add(doc)

	def find(self, doc: spacy.tokens.doc.Doc, options: dict = dict()):
		if "metrics" not in options:
			options = options.copy()
			options["metrics"] = self._metrics

		query = Query(self._vocab, doc, options)
		return self._corpus.find(query)
