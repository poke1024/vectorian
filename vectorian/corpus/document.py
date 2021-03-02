import json
import pyarrow as pa
import pandas as pd
import vectorian.core as core
import collections

from functools import lru_cache
from slugify import slugify


class SpansTable:
	_types = {
		'token_at': 'uint32',
		'n_tokens': 'uint16'
	}

	def __init__(self, loc_keys):
		self._loc = collections.defaultdict(list)
		self._loc_keys = loc_keys

	def extend(self, location, tokens):
		assert len(tokens) > 0
		loc = self._loc

		for k, v in zip(self._loc_keys, location):
			loc[k].append(v)

		if loc['token_at']:
			token_at = loc['token_at'][-1] + loc['n_tokens'][-1]
		else:
			token_at = 0

		loc['n_tokens'].append(len(tokens))
		loc['token_at'].append(token_at)

	def to_pandas(self):
		data = dict()
		for k, v in self._loc.items():
			dtype = SpansTable._types.get(k)
			if dtype is None:
				series = pd.Series(v)
			else:
				series = pd.Series(v, dtype=dtype)
			data[k] = series
		return pd.DataFrame(data)

	def to_arrow(self):
		return pa.Table.from_pandas(self.to_pandas())


class TokenTable:
	def __init__(self):
		self._idx = 0

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token

	def __len__(self):
		return len(self._token_idx)

	def extend(self, text, sent, tokens):
		last_idx = sent["start"]

		for token in tokens:
			idx = token["start"]
			self._idx += len(text[last_idx:idx])
			last_idx = idx

			token_text = text[token["start"]:token["end"]]
			self._token_idx.append(self._idx)
			self._token_len.append(len(token_text))

			self._token_pos.append(token["pos"])
			self._token_tag.append(token["tag"])

		self._idx += len(text[last_idx:sent["end"]])

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


def extract_token_str(table, text, normalizer):
	col_tok_idx = table["idx"].to_numpy()
	col_tok_len = table["len"].to_numpy()
	tokens = []
	for idx, len_ in zip(col_tok_idx, col_tok_len):
		t = normalizer(text[idx:idx + len_])
		tokens.append(t or "")
	return tokens


class Document:
	def __init__(self, json):
		self._json = json

	@staticmethod
	def load(path):
		with open(path, "r") as f:
			data = json.loads(f.read())
			data['metadata']['origin'] = path
			return Document(data)

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
	def metadata(self):
		return self._json['metadata']

	@property
	def unique_id(self):
		return self.metadata['unique_id']

	@property
	def caching_name(self):
		return slugify(self.unique_id)

	@property
	def origin(self):
		return self.metadata['origin']

	@property
	def title(self):
		return self.metadata['title']

	def prepare(self, session):
		return PreparedDocument(session, self._json)


class PreparedDocument:
	def __init__(self, session, json):
		self._session = session
		token_mapper = session.token_mapper('tagger')

		texts = []

		token_table = TokenTable()
		sentence_table = SpansTable(json['loc_keys'])

		partitions = json['partitions']
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
					t = token_mapper(t0)
					if t:
						sent_tokens.append(t)

				token_i = token_j

				sent_text = text[sent["start"]:sent["end"]]
				if sent_text.strip() and sent_tokens:
					token_table.extend(text, sent, sent_tokens)
					sentence_table.extend(loc, sent_tokens)
					texts.append(sent_text)

		self._text = "".join(texts)
		self._spans = {
			'sentence': sentence_table.to_arrow()
		}
		self._token_table = token_table.to_arrow()

		self._metadata = json['metadata']

	@property
	def text(self):
		return self._text

	@property
	def metadata(self):
		return self._metadata

	@property
	def unique_id(self):
		return self.metadata['unique_id']

	@property
	def caching_name(self):
		return slugify(self.unique_id)

	@property
	def n_tokens(self):
		return self._token_table.num_rows

	def n_spans(self, name):
		return self._spans[name].num_rows

	@lru_cache(16)
	def _spans(self, name):
		col_tok_idx = self._token_table["idx"]
		col_tok_len = self._token_table["len"]
		n_tokens = self._token_table.num_rows

		if name == "token":
			def get(i):
				start = col_tok_idx[i].as_py()
				end = start + col_tok_len[i].as_py()
				return self._text[start:end]

			return get
		else:
			col_token_at = self._spans[name].column('token_at')
			col_n_tokens = self._spans[name].column('n_tokens')

			def get(i):
				start = col_token_at[i].as_py()
				end = start + col_n_tokens[i].as_py()
				if end < n_tokens:
					i1 = col_tok_idx[end].as_py()
				else:
					i1 = col_tok_idx[end - 1].as_py() + col_tok_len[end - 1].as_py()
				return self._text[col_tok_idx[start].as_py():i1]

			return get

	def spans(self, name):
		get = self._spans(name)
		for i in range(self._spans[name].num_rows):
			yield get(i)

	def span(self, name, i):
		return self._spans(name)(i)

	def span_info(self, name, index):
		info = dict()
		if name == "token":
			return info  # FIXME
		table = self._spans[name]
		for k in table.column_names:
			col = table.column(k)
			info[k] = col[index].as_py()
		return info

	def to_core(self, index, vocab):
		return core.Document(
			index,
			vocab,
			self._spans,
			self._token_table,
			extract_token_str(
				self._token_table,
				self._text,
				self._session.token_mapper('tokenizer')),
			self._metadata)
