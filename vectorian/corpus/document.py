import json
import vectorian.core as core
import html
import numpy as np
import contextlib
import zipfile
import h5py
import uuid
import logging

from functools import lru_cache
from pathlib import Path
from cached_property import cached_property
from vectorian.embeddings import MaskedVectorsRef, ExternalMemoryVectorsRef


class TokenTable:
	def __init__(self, text, normalizers):
		self._text = text

		self._token_idx = []
		self._token_len = []
		self._token_pos = []  # pos_ from spacy's Token
		self._token_tag = []  # tag_ from spacy's Token
		self._token_str = []

		self._normalizers = normalizers
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

	def get(self):
		return self._text

	def close(self):
		self._f.close()
		
		
class ExternalSqliteText(Text):
	def __init__(self, doc_sql, unique_id):
		self._doc_sql = doc_sql
		self._unique_id = unique_id

	def get(self):
		cursor = self._doc_sql.cursor()
		try:
			cursor.execute("SELECT content FROM text WHERE unique_id=?", (self._unique_id,))
			text = cursor.fetchone()[0]
		finally:
			cursor.close()
		return text

	def close(self):
		pass


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


class Table:
	def __init__(self, get, count):
		self._col = {}
		self._get = get
		self._count = count

	def __len__(self):
		return self._count

	def __getitem__(self, k):
		data = self._col.get(k)
		if data is None:
			data = self._get(k)
			self._col[k] = data
		return data

	def __setitem__(self, k, v):
		self._col[k] = v


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

	def to_table(self):
		return Table(self._data.get, len(self))

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

	def __init__(self, hf):
		self._hf = hf
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

	def to_table(self):
		return Table(self._column, len(self))


class InternalMemoryDocumentStorage(DocumentStorage):
	def __init__(self, metadata, text, tokens, spans):
		self._metadata = metadata
		self._text = text
		self._tokens = InternalMemoryTokens(tokens)
		self._spans = spans

	@property
	def metadata(self):
		return self._metadata

	@contextlib.contextmanager
	def text(self, cache=None):
		yield InternalMemoryText(self._text)

	@contextlib.contextmanager
	def tokens(self):
		yield self._tokens

	@contextlib.contextmanager
	def spans(self):
		yield self._spans


def _spans_from_h5(hf):
	spans = dict()
	for name in hf.keys():
		data = dict()
		for k, v in hf[name].items():
			data[k] = v
		spans[name] = data

	return spans


class CorpusDocumentStorage(DocumentStorage):
	def __init__(self, doc_path, doc_sql, doc_group, unique_id):
		self._doc_path = doc_path
		self._doc_sql = doc_sql
		self._doc_group = doc_group
		self._unique_id = unique_id

	@cached_property
	def metadata(self):
		return json.loads(self._doc_group.attrs["metadata"])

	@contextlib.contextmanager
	def text(self, cache=None):
		if self._doc_sql is None:
			text = ExternalMemoryText(self._doc_path / "document.txt")
			try:
				yield text
			finally:
				text.close()
		else:
			yield ExternalSqliteText(self._doc_sql, self._unique_id)

	@contextlib.contextmanager
	def tokens(self):
		yield ExternalMemoryTokens(self._doc_group["tokens"])

	@contextlib.contextmanager
	def spans(self):
		yield _spans_from_h5(self._doc_group["spans"])


class ExternalMemoryDocumentStorage(DocumentStorage):
	def __init__(self, path):
		self._path = Path(path)

	@cached_property
	def metadata(self):
		with open("metadata.json", "r") as f:
			return json.loads(f.read())

	@contextlib.contextmanager
	def text(self, cache=None):
		text = ExternalMemoryText(self._path / "document.txt")
		try:
			yield text
		finally:
			text.close()

	@contextlib.contextmanager
	def tokens(self):
		with h5py.File(self._path / "tokens.h5", 'r') as hf:
			yield ExternalMemoryTokens(hf)

	@contextlib.contextmanager
	def spans(self):
		with h5py.File(self._path / "spans.h5", "r") as hf:
			yield _spans_from_h5(hf)


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
	def load_from_corpus(unique_id, doc_path, doc_sql, doc_group):
		contextual_embeddings = Document._load_embeddings(doc_path)
		return Document(CorpusDocumentStorage(
			doc_path, doc_sql, doc_group, unique_id), contextual_embeddings)

	@staticmethod
	def load_from_fs(path):
		path = Path(path)
		if path.suffix != ".document":
			raise ValueError(f"document path '{path}' must end in '.document'")

		contextual_embeddings = Document._load_embeddings(path)
		return Document(ExternalMemoryDocumentStorage(path), contextual_embeddings)

	def save_to_corpus(self, unique_id, doc_path, doc_sql, doc_group):
		doc_group.attrs["metadata"] = json.dumps(self._storage.metadata)

		with self._storage.tokens() as tokens:
			tokens.save_to_h5(doc_group.create_group("tokens"))

		with self._storage.spans() as spans:
			spans_group = doc_group.create_group("spans")
			for name, data in spans.items():
				g = spans_group.create_group(name)
				for k, v in data.items():
					g.create_dataset(k, data=v)

		with self._storage.text() as text:
			if doc_sql is None:
				doc_path.mkdir(exist_ok=True)

				with open(doc_path / "document.txt", "w") as f:
					f.write(text.get())
			else:
				with doc_sql:
					doc_sql.execute('''
						INSERT INTO text(unique_id, content) VALUES (?, ?)
						''', (unique_id, text.get()))

		self._save_embeddings(doc_path)

	def save_to_fs(self, path):
		path = Path(path)

		if not path.is_dir():
			raise ValueError(f"document path '{path}' needs to be a dir")
		if path.suffix != ".document":
			raise ValueError(f"document path '{path}' must end in '.document'")

		with open(path / "metadata.json", "w") as f:
			f.write(json.dumps(self._storage.metadata))

		with self._storage.tokens() as tokens:
			with h5py.File(path / "tokens.h5", "w") as hf:
				tokens.save_to_h5(hf)

		with self._storage.spans() as spans:
			with h5py.File(path / "spans.h5", "w") as hf:
				for name, data in spans.items():
					g = hf.create_group(name)
					for k, v in data.items():
						g.create_dataset(k, data=v)

		with self._storage.text() as text:
			with open(path / "document.txt", "w") as f:
				f.write(text.get())

		self._save_embeddings(path)

	@staticmethod
	def _load_embeddings(path):
		contextual_embeddings = dict()
		emb_path = path / "embeddings"
		emb_json_path = emb_path / "info.json"
		if emb_json_path.exists():
			with open(emb_json_path, "r") as f:
				emb_data = json.loads(f.read())

			for k, slug_name in emb_data.items():
				contextual_embeddings[k] = ExternalMemoryVectorsRef(emb_path / slug_name)
		return contextual_embeddings

	def _save_embeddings(self, path):
		if self._contextual_embeddings:
			emb_data = dict()

			emb_path = path / "embeddings"
			emb_path.mkdir(exist_ok=True, parents=True)

			for k, vectors in self._contextual_embeddings.items():
				unique_name = uuid.uuid1().hex
				assert not (emb_path / unique_name).exists()
				emb_data[k] = unique_name
				vectors.save(emb_path / unique_name)

			with open(emb_path / "info.json", "w") as f:
				f.write(json.dumps(emb_data))

	def to_json(self):
		return self._storage.metadata

	@property
	def structure(self):
		data = self._storage.metadata

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
	def origin(self):
		return self.metadata['origin']

	@property
	def title(self):
		return self.metadata['title']

	def prepare(self, corpus, flavor_cache, session):
		try:
			names = [v.factory.name for v in session.embeddings.values() if v.factory.is_contextual]
			contextual_embeddings = dict((k, self._contextual_embeddings[k]) for k in names)

			return PreparedDocument(
				corpus, flavor_cache, session, self, contextual_embeddings)
		except:
			logging.error(f"failed to prepare doc '{corpus.get_unique_id(self)}'")
			raise


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
	def __init__(self, corpus, flavor_cache, session, doc, contextual_embeddings):
		self._doc = doc
		self._session = session

		storage = doc.storage
		self._metadata = storage.metadata

		uid = corpus.get_unique_id(doc)
		flavor_record = flavor_cache.get(uid)

		token_mask = flavor_record.token_mask

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

		self._compiled = core.Document(
			corpus.get_doc_index(doc),
			session.vocab,
			self._spans,
			flavor_record.to_dict(),
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
		return self._doc.unique_id

	@contextlib.contextmanager
	def text(self):
		with self.storage.text() as text:
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
