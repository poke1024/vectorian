import json
import vectorian.core as core
import html
import numpy as np
import contextlib
import zipfile
import h5py
import uuid

from functools import lru_cache
from slugify import slugify
from pathlib import Path
from vectorian.embeddings import MaskedVectorsRef, ExternalMemoryVectorsRef


class Text:
	def get(self):
		raise NotImplementedError()

	def close(self):
		raise NotImplementedError()


class InternalMemoryText(Text):
	def __init__(self, text):
		self._text = text

	def get(self):
		return self._text

	def close(self):
		pass


class ExternalMemoryText(Text):
	def __init__(self, path):
		self._f = open(path, "r")
		self._text = self._f.read()
		# FIXME

	def get(self):
		return self._text

	def close(self):
		self._f.close()


def convert_idx_len_to_utf8(text, idx, len):
	pass  # FIXME


class TokenTable:
	def __init__(self, text, normalizers):
		self._text = text

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token
		self._token_str = []

		self._make_text = normalizers['token'].token_to_text
		self._norm_text = normalizers['text'].to_callable()

	def __len__(self):
		return len(self._token_idx)

	def add(self, token):
		norm_text = self._norm_text(self._make_text(self._text, token)) or ""
		if len(norm_text.strip()) > 0:
			self._token_idx.append(token["start"])
			self._token_len.append(token["end"] - token["start"])

			self._token_pos.append(token.get("pos", "") or "")
			self._token_tag.append(token.get("tag", "") or "")

			self._token_str.append(norm_text)
			return True
		else:
			return False

	def to_dict(self):
		max_len = np.iinfo(np.uint8).max
		if any(x > max_len for x in self._token_len):
			raise RuntimeError(f"token len > {max_len} is not supported")

		return {
			'str': self._token_str,
			'idx': np.array(self._token_idx, dtype=np.uint32),
			'len': np.array(self._token_len, dtype=np.uint8),
			'pos': self._token_pos,
			'tag': self._token_tag
		}


class Lengths:
	def __init__(self, start, end):
		self._start = start
		self._end = end

	def __len__(self):
		return self._start.shape[0]

	def __getitem__(self, i):
		return self._end[i] - self._start[i]


def xspan(idxs, lens, i0, window_size, window_step):
	i = i0 * window_step
	start = idxs[i]
	j = i + window_size
	if j <= len(idxs) - 1:
		end = idxs[j]
	else:
		end = idxs[-1] + lens[-1]
	return start, end


class DocumentStorage:
	@property
	def metadata(self):
		raise NotImplementedError()


class Tokens:
	pass


class InternalMemoryTokens(Tokens):
	class TokenData:
		def __init__(self, data, index):
			self._data = data
			self._index = index

		def copy(self):
			i = self._index
			return dict((k, v[i]) for k, v in self._data.items())

		def get(self, key, default=None):
			array = self._data.get(key)
			return default if array is None else array[self._index]

		def __getitem__(self, key):
			return self.get(key)

	def __init__(self, tokens):
		self._tokens = tokens
		self._data = dict((k, v['data']) for k, v in tokens.items())

	def __len__(self):
		return len(self._tokens['start']['data'])

	def __iter__(self):
		for i in range(len(self)):
			yield InternalMemoryTokens.TokenData(self._data, i)

	def save_to_h5(self, hf):
		for k, v in self._tokens.items():
			dtype = v['dtype']
			data = v['data']
			if dtype == 'int':
				hf.create_dataset(k, data=data)
			elif dtype == 'enum':
				mapping = dict((k, i) for i, k in enumerate(set(data)))
				assert len(mapping) <= 0xff
				dt = h5py.enum_dtype(mapping, basetype=np.uint8)
				hf.create_dataset(k, dtype=dt, data=[mapping[x] for x in data])
			elif dtype == 'str':
				dt = h5py.string_dtype(encoding='utf8')
				strs = ['' if x is None else str(x) for x in data]
				hf.create_dataset(k, dtype=dt, data=[s.encode("utf8") for s in strs], compression='lzf')
			else:
				raise ValueError(dtype)


class ExternalMemoryTokens(Tokens):
	class TokenData:
		def __init__(self, hf, get, index):
			self._hf = hf
			self._get = get
			self._index = index

		def copy(self):
			i = self._index
			return dict((k, self._get(k)[i]) for k in self._hf.keys())

		def items(self):
			return self.copy().items()

		def get(self, key, default=None):
			array = self._get(key)
			return default if array is None else array[self._index]

		def __getitem__(self, key):
			return self.get(key)

	def __init__(self, path):
		self._hf = h5py.File(path, 'r')
		self._cache = dict()

	def _column(self, k):
		cached_data = self._cache.get(k)
		if cached_data is not None:
			return cached_data

		dset = self._hf.get(k)
		if dset is not None:
			if h5py.check_string_dtype(dset.dtype):
				py_data = dset.asstr()
			else:
				enum_dict = h5py.check_enum_dtype(dset.dtype)
				if enum_dict:
					inv_enum_dict = dict((i, k) for k, i in enum_dict.items())
					py_data = [inv_enum_dict[x] for x in np.array(dset)]
				else:
					py_data = np.array(dset)
		else:
			py_data = None

		self._cache[k] = py_data
		return py_data

	def __len__(self):
		return self._hf['start'].shape[0]

	def __iter__(self):
		get = self._column
		for i in range(len(self)):
			yield ExternalMemoryTokens.TokenData(self._hf, get, i)

	def close(self):
		self._hf.close()


class InternalMemoryDocumentStorage(DocumentStorage):
	def __init__(self, json, text, tokens, spans):
		self._json = json
		self._text = text
		self._tokens = InternalMemoryTokens(tokens)
		self._spans = spans

	@property
	def metadata(self):
		return self._json['metadata']

	@contextlib.contextmanager
	def json(self):
		yield self._json

	@contextlib.contextmanager
	def text(self, session):
		yield InternalMemoryText(self._text)

	@contextlib.contextmanager
	def tokens(self):
		yield self._tokens

	@contextlib.contextmanager
	def spans(self):
		yield self._spans


class ExternalMemoryDocumentStorage(DocumentStorage):
	def __init__(self, path):
		self._path = Path(path)
		self._metadata = None

	def _load_metadata(self, zf):
		data = json.loads(zf.read('data.json'))
		self._metadata = data['metadata']
		self._metadata['origin'] = str(self._path)

	@property
	def metadata(self):
		if self._metadata is None:
			with zipfile.ZipFile(self._path.with_suffix('.zip'), 'r') as zf:
				self._load_metadata(zf)
		return self._metadata

	@contextlib.contextmanager
	def json(self):
		with zipfile.ZipFile(self._path.with_suffix('.zip'), 'r') as zf:
			if self._metadata is None:
				self._load_metadata(zf)
			yield json.loads(zf.read('data.json'))

	@contextlib.contextmanager
	def text(self, session):
		text = ExternalMemoryText(self._path.with_suffix('.txt'))
		try:
			yield text
		finally:
			text.close()

	@contextlib.contextmanager
	def tokens(self):
		tokens = ExternalMemoryTokens(self._path.with_suffix(".tok.h5"))

		try:
			yield tokens
		finally:
			tokens.close()

	@contextlib.contextmanager
	def spans(self):
		with h5py.File(self._path.with_suffix(".spn.h5"), "r") as hf:
			spans = dict()
			for name in hf.keys():
				data = dict()
				for k, v in hf[name].items():
					data[k] = v
				spans[name] = data
			yield spans


class Document:
	def __init__(self, storage, contextual_embeddings=None):
		self._storage = storage
		self._contextual_embeddings = contextual_embeddings or {}

	@property
	def storage(self):
		return self._storage

	@property
	def contextual_embeddings(self):
		return self._contextual_embeddings

	def has_contextual_embedding(self, name):
		return name in self._contextual_embeddings

	@staticmethod
	def load(path):
		path = Path(path)

		contextual_embeddings = dict()
		emb_path = path.with_suffix(".embeddings")
		emb_json_path = emb_path / "info.json"
		if emb_json_path.exists():
			with open(emb_json_path, "r") as f:
				emb_data = json.loads(f.read())

			for k, slug_name in emb_data.items():
				contextual_embeddings[k] = ExternalMemoryVectorsRef(emb_path / slug_name)

		return Document(ExternalMemoryDocumentStorage(path), contextual_embeddings)

	def save(self, path):
		path = Path(path)

		with self._storage.json() as data:
			with zipfile.ZipFile(path.with_suffix(".zip"), "w") as zf:
				zf.writestr("data.json", json.dumps(data))

		with self._storage.tokens() as tokens:
			with h5py.File(path.with_suffix(".tok.h5"), "w") as hf:
				tokens.save_to_h5(hf)

		with self._storage.spans() as spans:
			with h5py.File(path.with_suffix(".spn.h5"), "w") as hf:
				for name, data in spans.items():
					g = hf.create_group(name)
					for k, v in data.items():
						g.create_dataset(k, data=v)

		with self._storage.text(None) as text:
			with open(path.with_suffix(".txt"), "w") as f:
				f.write(text.get())

		if self._contextual_embeddings:
			emb_data = dict()

			emb_path = path.with_suffix(".embeddings")
			emb_path.mkdir(exist_ok=True)

			for k, vectors in self._contextual_embeddings.items():
				unique_name = uuid.uuid1().hex
				assert not (emb_path / unique_name).exists()
				emb_data[k] = unique_name
				vectors.save(emb_path / unique_name)

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

	def prepare(self, session, doc_index):
		names = [v.factory.name for v in session.embeddings.values() if v.factory.is_contextual]
		contextual_embeddings = dict((k, self._contextual_embeddings[k]) for k in names)

		return PreparedDocument(
			self, session, doc_index, contextual_embeddings)


class Token:
	_css = 'background:	#F5F5F5; border-radius:0.25em;'
	_html_template = '<span style="{style}">{text}</span>'

	def __init__(self, doc, table, index):
		self._doc = doc
		self._table = table
		self._index = index

	@property
	def doc(self):
		return self._doc

	@property
	def index(self):
		return self._index

	def to_slice(self):
		offset = self._table["idx"][self._index]
		return slice(offset, offset + self._table["len"][self._index])

	@property
	def text(self):
		with self._doc.text() as text:
			return text.get()[self.to_slice()]

	def _repr_html_(self):
		return Token._html_template.format(
			style=Token._css, text=html.escape(self.text))


class Span:
	def __init__(self, doc, table, start, end):
		self._doc = doc
		self._table = table
		self._start = start
		self._end = end

	@property
	def start(self):
		return self._start

	@property
	def end(self):
		return self._end

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

		if len(col_tok_idx) == 0:
			return ""

		i0, i1 = xspan(
			col_tok_idx, col_tok_len, self._start,
			self._end - self._start, 1)

		with self._doc.text() as text:
			return text.get()[i0:i1]

	def _repr_html_(self):
		tokens = []
		for i in range(self._end - self._start):
			tokens.append(self[i]._repr_html_())
		return " ".join(tokens)


class PreparedDocument:
	def __init__(self, doc, session, doc_index, contextual_embeddings):
		self._doc = doc
		storage = doc.storage

		self._session = session

		token_mapper = session.normalizer('token').token_to_token

		with storage.tokens() as tokens:
			with storage.text(self._session) as text:
				token_table = TokenTable(text.get(), self._session.normalizers)

				token_mask = np.zeros((len(tokens),), dtype=np.bool)

				for i, token in enumerate(tokens):
					t = token_mapper(token)
					if t:
						token_mask[i] = token_table.add(t)

		with storage.spans() as spans:
			reindex = np.cumsum(np.concatenate(([False], token_mask)), dtype=np.int32)

			self._spans = dict()
			for name, data in spans.items():
				new_data = dict((k, np.array(v)) for k, v in data.items())
				new_data['start'] = reindex[new_data['start']]
				new_data['end'] = reindex[new_data['end']]
				self._spans[name] = new_data

		self._contextual_embeddings = dict(
			(k, MaskedVectorsRef(v, token_mask)) for k, v in contextual_embeddings.items())

		self._metadata = storage.metadata

		self._compiled = core.Document(
			doc_index,
			session.vocab,
			self._spans,
			token_table.to_dict(),
			self._metadata,
			self._contextual_embeddings)

		self._tokens = self._compiled.tokens

	def cache_contextual_embeddings(self):
		self._contextual_embeddings = dict(
			(k, v.to_internal_memory()) for k, v in self._contextual_embeddings.items())

	@property
	def doc(self):
		return self._doc

	@property
	def session(self):
		return self._session

	@property
	def storage(self):
		return self._doc.storage

	@property
	def contextual_embeddings(self):
		return self._doc.contextual_embeddings

	def _save_tokens(self, path):
		with open(path, "w") as f:
			col_tok_idx = self._tokens["idx"]
			col_tok_len = self._tokens["len"]
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

	@contextlib.contextmanager
	def text(self):
		with self.storage.text(self._session) as text:
			yield text

	def token(self, i):
		return Token(self, self._tokens, i)

	@property
	def n_tokens(self):
		return self.compiled.n_tokens

	def n_spans(self, partition):
		if partition.level == "token":
			return self.n_tokens
		else:
			n = self._spans[partition.level]['start'].shape[0]
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
				return Span(self, self._tokens, pos, pos + window_size)

			return get
		else:
			col_start = self._spans[name]['start']
			col_len = Lengths(col_start, self._spans[name]['end'])

			def get(i):
				start, end = xspan(
					col_start, col_len, i,
					window_size, window_step)

				return Span(self, self._tokens, start, end)

			return get

	def spans(self, partition):
		get = self._spans_getter(partition)
		for i in range(self.n_spans(partition)):
			yield get(i)

	def span(self, partition, index):
		return self._spans_getter(partition)(index)

	def span_info(self, partition, slice_id):
		# note that slice_id is already a multiple of
		# partition.window_step, i.e. the final offset
		if partition.level == "token":
			return {
				'start': slice_id,
				'end': slice_id + partition.window_size
			}
		else:
			table = self._spans[partition.level]
			info = dict((k, int(v[slice_id])) for k, v in table.items())
			return info

	@property
	def compiled(self):
		return self._compiled
