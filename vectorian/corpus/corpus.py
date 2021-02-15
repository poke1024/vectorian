import vectorian.core as core


class Corpus:
	def __init__(self, vocab):
		self._vocab = vocab
		self._docs = []

	def add(self, doc):
		self._docs.append(
			doc.to_core(len(self._docs), self._vocab)
		)

	def find(self, query):
		results = None
		c_query = query.to_core()
		for doc in self._docs:
			x = doc.find(c_query)
			if results is None:
				results = x
			else:
				results.extend(x)
		return results
