import json
import pyarrow as pa
import pandas as pd
import vectorian.core as core


class DefaultTokenFilter:
	_pos = {
		'PROPN': 'NOUN'
	}

	_tag = {
		'NNP': 'NN',
		'NNPS': 'NNS',
	}

	def __call__(self, t):
		if t["pos"] == "PUNCT":
			return False

		# spaCy is very generous with labeling things as PROPN,
		# which really breaks pos_mimatch_penalty often. we re-
		# classify PROPN as NOUN.
		t_new = t.copy()
		t_new["pos"] = self._pos.get(t["pos"], t["pos"])
		t_new["tag"] = self._tag.get(t["tag"], t["tag"])
		return t_new


class LocationTable:
	def __init__(self):
		self._loc = [[] for _ in range(5)]

	def extend(self, location, tokens):
		loc = self._loc

		loc[0].append(location['bk'])
		loc[1].append(location['ch'])
		loc[2].append(location['sp'])
		loc[3].append(location['l'])

		loc[4].append(len(tokens))

	def to_pandas(self):
		loc = self._loc
		return pd.DataFrame({
			'book': pd.Series(loc[0], dtype='int8'),
			'chapter': pd.Series(loc[1], dtype='int8'),
			'speaker': pd.Series(loc[2], dtype='int8'),
			'location': pd.Series(loc[3], dtype='uint16'),
			'n_tokens': pd.Series(loc[4], dtype='uint16')})

	def to_arrow(self):
		return pa.Table.from_pandas(self.to_pandas())


class TokenTable:
	def __init__(self):
		self._utf8_idx = 0

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token

	def __len__(self):
		return len(self._token_idx)

	def extend(self, text, tokens):
		last_idx = 0

		for token in tokens:
			idx = token["start"]
			self._utf8_idx += len(text[last_idx:idx].encode('utf8'))
			last_idx = idx

			token_text = text[token["start"]:token["end"]]
			self._token_idx.append(self._utf8_idx)
			self._token_len.append(len(token_text.encode('utf8')))

			self._token_pos.append(token["pos"])
			self._token_tag.append(token["tag"])

		self._utf8_idx += len(text[last_idx:].encode('utf8'))

	def to_pandas(self):
		return pd.DataFrame({
			'idx': pd.Series(self._token_idx, dtype='uint32'),
			'len': pd.Series(self._token_len, dtype='uint8'),
			'pos': pd.Series(self._token_pos, dtype='category'),
			'tag': pd.Series(self._token_tag, dtype='category')})

	def to_arrow(self):
		tokens_table_data = [
			pa.array(self._token_idx, type=pa.uint32()),
			pa.array(self._token_len, type=pa.uint8()),
			pa.array(self._token_pos, type=pa.string()),
			pa.array(self._token_tag, type=pa.string())
		]

		return pa.Table.from_arrays(
			tokens_table_data,
			['idx', 'len', 'pos', 'tag'])


class Document:
	def __init__(self, json):
		self._json = json

	@staticmethod
	def load(path):
		with open(path, "r") as f:
			return Document(json.loads(f.read()))

	def save(self, path):
		with open(path, "w") as f:
			f.write(json.dumps(self._json, indent=4, sort_keys=True))

	@property
	def title(self):
		return self._json['title']

	def to_core(self, index, vocab, filter=DefaultTokenFilter()):
		texts = []

		token_table = TokenTable()
		location_table = LocationTable()

		partitions = self._json['partitions']
		for partition in partitions:
			text = partition["text"]
			tokens = partition["tokens"]
			loc = partition["loc"]

			p_tokens = []
			for sent in partition["sents"]:
				for t0 in tokens[sent["start"]:sent["end"]]:
					t = filter(t0)
					if t:
						p_tokens.append(t)

			token_table.extend(text, p_tokens)
			location_table.extend(loc, p_tokens)

			texts.append(text)

		return core.Document(
			index,
			vocab,
			"".join(texts).encode("utf8"),
			location_table.to_arrow(),
			token_table.to_arrow(),
			{
				'author': self._json['author'],
				'title': self._json['title']
			},
			"")
