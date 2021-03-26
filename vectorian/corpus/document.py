import json
import pyarrow as pa
import pandas as pd
import vectorian.core as core
import collections
import html
import numpy as np
import contextlib
import zipfile

from functools import lru_cache
from slugify import slugify
from pathlib import Path
from vectorian.embeddings import MaskedVectorsRef, OnDiskVectorsRef


class SpansTable:
	_types = {
		'token_at': 'uint32',
		'n_tokens': 'uint16'
	}

	def __init__(self, loc_keys):
		self._loc = collections.defaultdict(list)
		self._loc_keys = loc_keys

	def extend(self, location, n_tokens):
		if n_tokens < 1:
			return

		loc = self._loc

		for k, v in zip(self._loc_keys, location):
			loc[k].append(v)

		if loc['token_at']:
			token_at = loc['token_at'][-1] + loc['n_tokens'][-1]
		else:
			token_at = 0

		loc['n_tokens'].append(n_tokens)
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
	def __init__(self, normalizers):
		self._idx = 0

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token
		self._token_str = []

		self._make_text = normalizers['token'].token_to_text
		self._norm_text = normalizers['text'].to_callable()

	@property
	def normalized_tokens(self):
		return self._token_str

	def __len__(self):
		return len(self._token_idx)

	def advance(self, n):
		self._idx += n

	def extend(self, text, sent, tokens):
		last_idx = sent["start"]
		picked = []

		for i, token in enumerate(tokens):
			idx = token["start"]
			self._idx += len(text[last_idx:idx])
			last_idx = idx

			norm_text = self._norm_text(self._make_text(text, token)) or ""
			if len(norm_text.strip()) > 0:
				self._token_idx.append(self._idx)
				self._token_len.append(len(text[token["start"]:token["end"]]))

				self._token_pos.append(token["pos"])
				self._token_tag.append(token["tag"])

				self._token_str.append(norm_text)
				picked.append(i)

		self._idx += len(text[last_idx:sent["end"]])

		return picked

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


def xspan(idxs, lens, i0, window_size, window_step):
	i = i0 * window_step
	start = idxs[i].as_py()
	j = i + window_size
	if j <= len(idxs) - 1:
		end = idxs[j].as_py()
	else:
		end = idxs[-1].as_py() + lens[-1].as_py()
	return start, end


class DocumentStorage:
	@property
	def metadata(self):
		raise NotImplementedError()


class InMemoryDocumentStorage(DocumentStorage):
	def __init__(self, json):
		self._json = json

	@property
	def metadata(self):
		return self._json['metadata']

	@contextlib.contextmanager
	def json(self):
		yield self._json


class OnDiskDocumentStorage(DocumentStorage):
	def __init__(self, path):
		self._path = Path(path).with_suffix('.zip')
		self._metadata = None

	@property
	def metadata(self):
		if self._metadata is None:
			with zipfile.ZipFile(self._path, 'r') as zf:
				self._metadata = json.loads(zf.read('metadata.json'))
		return self._metadata

	@contextlib.contextmanager
	def json(self):
		with zipfile.ZipFile(self._path, 'r') as zf:
			if self._metadata is None:
				self._metadata = json.loads(zf.read('metadata.json'))
			yield json.loads(zf.read('data.json'))


class Document:
	def __init__(self, storage, contextual_embeddings=None):
		self._storage = storage
		self._contextual_embeddings = contextual_embeddings or {}

	@property
	def contextual_embeddings(self):
		return self._contextual_embeddings

	def has_contextual_embedding(self, name):
		return name in self._contextual_embeddings

	@staticmethod
	def load(path):
		path = Path(path)

		with open(path.with_suffix(".json"), "r") as f:
			data = json.loads(f.read())
			data['metadata']['origin'] = path

		contextual_embeddings = dict()
		emb_path = path.with_suffix(".embeddings")
		emb_json_path = emb_path / "info.json"
		if emb_json_path.exists():
			with open(emb_json_path, "r") as f:
				emb_data = json.loads(f.read())

			for k, slug_name in emb_data.items():
				contextual_embeddings[k] = OnDiskVectorsRef(emb_path / slug_name)

		return Document(InMemoryDocumentStorage(data), contextual_embeddings)

	def save(self, path):
		path = Path(path)

		with self._storage.json() as data:
			with zipfile.ZipFile(path.with_suffix(".zip"), "w") as zf:
				zf.writestr("metadata.json", json.dumps(self.metadata))
				zf.writestr("data.json", json.dumps(data))

		if self._contextual_embeddings:
			emb_data = dict()

			emb_path = path.with_suffix(".embeddings")
			emb_path.mkdir(exist_ok=True)

			for k, vectors in self._contextual_embeddings.items():
				slug_name = slugify(k)
				emb_data[k] = slug_name
				vectors.save(emb_path / slug_name)

			with open(emb_path / "info.json", "w") as f:
				f.write(json.dumps(emb_data))

	def to_json(self):
		with self._storage.json() as data:
			r = data
		return r

	@property
	def structure(self):
		with self._storage.json() as data:
			lines = []
			for i, p in enumerate(data["partitions"]):
				text = p["text"]
				lines.append(f"partition {i + 1}:")
				for j, sent in enumerate(p["sents"]):
					lines.append(f"  sentence {j + 1}:")
					lines.append("    " + text[sent["start"]:sent["end"]])
			return "\n".join(lines)

	@property
	def metadata(self):
		return self._storage.metadata

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
		names = [e.name for e in session.embeddings if e.is_contextual]
		contextual_embeddings = dict((k, self._contextual_embeddings[k]) for k in names)

		return PreparedDocument(
			session, self._storage, contextual_embeddings)


class Token:
	_css = 'background:	#F5F5F5; border-radius:0.25em;'
	_html_template = '<span style="{style}">{text}</span>'

	def __init__(self, doc, table, index):
		self._doc = doc
		self._table = table
		self._index = index

	def to_slice(self):
		offset = self._table["idx"][self._index].as_py()
		return slice(offset, offset + self._table["len"][self._index].as_py())

	@property
	def text(self):
		return self._doc.text[self.to_slice()]

	def _repr_html_(self):
		return Token._html_template.format(
			style=Token._css, text=html.escape(self.text))


class Span:
	def __init__(self, doc, table, start, end):
		self._doc = doc
		self._table = table
		self._start = start
		self._end = end

	def __iter__(self):
		for i in range(self._end - self._start):
			yield self[i]

	def __getitem__(self, i):
		n = self._end - self._start
		if i < 0 or i >= n:
			raise IndexError(f'{i} not in [0, {n}[')
		return Token(self._doc, self._table, self._start + i)

	def __len__(self):
		return self._end - self._start

	@property
	def text(self):
		col_tok_idx = self._table["idx"]
		col_tok_len = self._table["len"]

		i0, i1 = xspan(
			col_tok_idx, col_tok_len, self._start,
			self._end - self._start, 1)

		return self._doc.text[i0:i1]

	def _repr_html_(self):
		tokens = []
		for i in range(self._end - self._start):
			tokens.append(self[i]._repr_html_())
		return " ".join(tokens)


class PreparedDocument:
	def __init__(self, session, storage, contextual_embeddings):
		self._session = session

		token_mapper = session.normalizer('token').token_to_token

		texts = []

		with storage.json() as json:

			token_table = TokenTable(self._session.normalizers)
			sentence_table = SpansTable(json['loc_keys'])

			token_count = sum(len(p['tokens']) for p in json['partitions'])
			token_mask = np.zeros((token_count,), dtype=np.bool)
			token_mask_offset = 0

			for partition_i, partition in enumerate(json['partitions']):
				text = partition["text"]
				tokens = partition["tokens"]
				sents = partition["sents"]
				loc = partition["loc"]

				token_i = 0
				last_sent_end = None
				for sent in sents:
					sent_tokens = []
					sent_text = text[sent["start"]:sent["end"]]

					if last_sent_end is not None and sent["start"] > last_sent_end:
						s = text[last_sent_end:sent["start"]]
						token_table.advance(len(s))
						texts.append(s)

					if sent["start"] > tokens[token_i]["start"]:
						raise RuntimeError(
							f"unexpected sentence start {sent['start']} vs. {token_i}, "
							f"partition={partition_i}, tokens={tokens}, sents={sents}")

					token_j = token_i + 1
					while token_j < len(tokens):
						if tokens[token_j]["start"] >= sent["end"]:
							break
						token_j += 1

					for i in range(token_i, token_j):
						t = token_mapper(tokens[i])
						if t:
							sent_tokens.append((i, t))

					token_i = token_j

					if sent_text.strip() and sent_tokens:
						picked = token_table.extend(text, sent, [t for _, t in sent_tokens])
						sentence_table.extend(loc, len(picked))
						texts.append(sent_text)
						picked_indices = np.array([sent_tokens[i][0] for i in picked], dtype=np.int32)
						token_mask[token_mask_offset + picked_indices] = True

					last_sent_end = sent["end"]

				token_mask_offset += len(tokens)

		assert np.sum(token_mask) == len(token_table)

		self._text = "".join(texts)
		self._spans = {
			'sentence': sentence_table.to_arrow()
		}
		self._token_table = token_table.to_arrow()
		self._token_str = token_table.normalized_tokens

		self._contextual_embeddings = dict(
			(k, MaskedVectorsRef(v, token_mask)) for k, v in contextual_embeddings.items())

		self._metadata = storage.metadata

	def _save_tokens(self, path):
		with open(path, "w") as f:
			col_tok_idx = self._token_table["idx"].to_numpy()
			col_tok_len = self._token_table["len"].to_numpy()
			for idx, len_ in zip(col_tok_idx, col_tok_len):
				f.write('"' + self._text[idx:idx + len_] + '"\n')

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
	def text(self):
		return self._text

	def token(self, i):
		return Token(self, self._token_table, i)

	@property
	def n_tokens(self):
		return self._token_table.num_rows

	def n_spans(self, partition):
		n = self._spans[partition.level].num_rows
		k = n // partition.window_step
		if (k * partition.window_step) < n:
			k += 1
		return k

	def _spans_getter(self, p):
		return self._cached_spans(
			p.level, p.window_size, p.window_step)

	@lru_cache(16)
	def _cached_spans(self, name, window_size, window_step):
		if name == "token":
			def get(i):
				pos = i * window_step
				return Span(self, self._token_table, pos, pos + window_size)

			return get
		else:
			col_token_at = self._spans[name].column('token_at')
			col_n_tokens = self._spans[name].column('n_tokens')

			def get(i):
				start, end = xspan(
					col_token_at, col_n_tokens, i,
					window_size, window_step)

				return Span(self, self._token_table, start, end)

			return get

	def spans(self, partition):
		get = self._spans_getter(partition)
		for i in range(self.n_spans(partition)):
			yield get(i)

	def span(self, partition, index):
		return self._spans_getter(partition)(index)

	def span_info(self, partition, index):
		info = dict()
		if partition.level == "token":
			return info  # FIXME
		table = self._spans[partition.level]
		i = index * partition.window_step
		for k in table.column_names:
			col = table.column(k)
			info[k] = col[i].as_py()
		return info

	def to_core(self, index, vocab):
		return core.Document(
			index,
			vocab,
			self._spans,
			self._token_table,
			self._token_str,
			self._metadata,
			self._contextual_embeddings)
