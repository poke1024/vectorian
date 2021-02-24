import json
import pyarrow as pa
import pandas as pd
import vectorian.core as core


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

	def free_up_memory(self):
		self._json = {
			'unique_id': self._json['unique_id'],
			'origin': self._json['origin'],
			'author': self._json['author'],
			'title': self._json['title']
		}

	@staticmethod
	def load(path):
		with open(path, "r") as f:
			doc = Document(json.loads(f.read()))
			doc['origin'] = path
			return doc

	def save(self, path):
		with open(path, "w") as f:
			f.write(json.dumps(self._json, indent=4, sort_keys=True))

	def to_json(self):
		return self._json

	@property
	def structure(self):
		lines = []
		for i, p in enumerate(self._json["partitions"]):
			text = p["text"]
			lines.append(f"partition {i + 1}:")
			for j, sent in enumerate(p["sents"]):
				lines.append(f"  sentence {j + 1}:")
				lines.append("    " + text[sent["start"]:sent["end"]])
		return "\n".join(lines)

	@property
	def unique_id(self):
		return self._json['unique_id']

	@property
	def origin(self):
		return self._json['origin']

	@property
	def title(self):
		return self._json['title']

	def to_core(self, index, vocab, filter_):
		texts = []

		token_table = TokenTable()
		location_table = LocationTable()

		partitions = self._json['partitions']
		for partition_i, partition in enumerate(partitions):
			text = partition["text"]
			tokens = partition["tokens"]
			sents = partition["sents"]
			loc = partition["loc"]

			token_i = 0
			for sent in sents:
				sent_tokens = []

				if sent["start"] > tokens[token_i]["start"]:
					raise RuntimeError(
						f"unexpected sentence start {sent['start']} vs. {token_i}, "
						f"partition={partition_i}, tokens={tokens}, sents={sents}")

				token_j = token_i + 1
				while token_j < len(tokens):
					if tokens[token_j]["start"] >= sent["end"]:
						break
					token_j += 1

				for t0 in tokens[token_i:token_j]:
					t = filter_(t0)
					if t:
						sent_tokens.append(t)

				token_i = token_j

				sent_text = text[sent["start"]:sent["end"]]

				token_table.extend(sent_text, sent_tokens)
				location_table.extend(loc, sent_tokens)

				texts.append(sent_text)

		return core.Document(
			self,
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
