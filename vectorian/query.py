import vectorian.core as core

from vectorian.corpus.document import TokenTable


class Query:
	def __init__(self, vocab, doc, options):
		self._vocab = vocab
		self._doc = doc
		self._options = options

	def to_core(self):
		tokens = self._doc.to_json()["tokens"]

		if self._options.get('ignore_determiners'):
			tokens = [t for t in tokens if t["pos"] != "DET"]

		token_table = TokenTable()
		token_table.extend(self._doc.text, tokens)

		return core.Query(
			self._vocab,
			self._doc.text,
			token_table.to_arrow(),
			**self._options)
