import concurrent.futures
import numpy as np
import h5py
import collections
import enum
import threading
import sqlite3
import contextlib
import tempfile
import shutil
import datetime
import uuid
import json

from vectorian.corpus.document import Document
from vectorian.tqdm import tqdm
from pathlib import Path


class FlavorRecord:
	def __init__(self, doc_group, mappings):
		self._doc_group = doc_group
		self._mappings = mappings

	@property
	def token_mask(self):
		return np.array(self._doc_group["token_mask"])

	def _unmap(self, attr):
		xs = np.array(self._doc_group[attr])
		m = self._mappings[attr]
		return [m[i] for i in xs]

	def to_dict(self):
		span = self._doc_group["span"]

		max_len = np.iinfo(np.uint8).max
		if np.any(span[:, 1] > max_len):
			raise RuntimeError(f"token len > {max_len} is not supported")

		return {
			'str': self._unmap('str'),
			'idx': np.array(span[:, 0], dtype=np.uint32),
			'len': np.array(span[:, 1], dtype=np.uint8),
			'pos': self._unmap('pos'),
			'tag': self._unmap('tag')
		}


class FlavorCache:
	def __init__(self, root):
		self._root = root
		self._documents_group = root['documents']
		self._mappings_group = root['mappings']

		self._mappings = {}
		for k in self._mappings_group.keys():
			dset = self._mappings_group[k]
			n = dset.shape[0]
			self._mappings[k] = [str(dset[i]) for i in range(n)]

	def get(self, unique_id):
		return FlavorRecord(
			self._documents_group[unique_id],
			self._mappings)


class FlavorBuilder:
	class Stage(enum.Enum):
		PREFLIGHT = 0
		ADD = 1

	def __init__(self, root, normalizers):
		self._root = root
		self._documents_group = root.create_group('documents')
		self._mappings_group = root.create_group('mappings')

		self._normalizers = normalizers
		self._make_text = normalizers['token'].token_to_text
		self._token_to_token = normalizers['token'].token_to_token_many
		self._norm_text = normalizers['text'].to_callable()

		self._mappings = collections.defaultdict(dict)
		self._enums = {}

		self._stage = None

	@staticmethod
	def make(root, normalization, docs):
		builder = FlavorBuilder(root, normalization.normalizers)

		with tqdm(total=2 * len(docs), desc=f"Adding Normalization '{normalization.name}'") as pbar:
			for stage in (FlavorBuilder.Stage.PREFLIGHT, FlavorBuilder.Stage.ADD):
				builder.set_stage(stage)
				for unique_id, doc in docs.items():
					builder.add(unique_id, doc)
					pbar.update(1)

	def _add_mappings(self, attr, value):
		if not isinstance(value, str):
			raise ValueError(f"expected str, got '{value}'")
		m = self._mappings[attr]
		if value not in m:
			m[value] = len(m)

	def _add_mapping(self, attr):
		dt = h5py.string_dtype(encoding='utf8')
		mapping = self._mappings[attr]

		self._mappings_group.create_dataset(
			attr,
			data=[x.encode("utf8") for x, _ in sorted(mapping.items(), key=lambda x: x[1])],
			dtype=dt)

	def set_stage(self, stage: Stage):
		if stage == FlavorBuilder.Stage.ADD:
			for attr in self._mappings.keys():
				self._add_mapping(attr)
		self._stage = stage

	def _add(self, unique_id, doc, text, table, base_mask):
		texts = list(map(
			self._norm_text,
			self._normalizers['token'].token_to_text_many(text, table)))
		token_mask = np.logical_and(
			base_mask,
			np.array([x and len(x.strip()) > 0 for x in texts], dtype=np.bool))

		def coalesce(x):
			if x is None:
				return [None] * token_mask.shape[0]
			else:
				return x

		pos_data = coalesce(table["pos"])
		tag_data = coalesce(table["tag"])

		data = {
			'pos': [(x or "") for i, x in enumerate(pos_data) if token_mask[i]],
			'tag': [(x or "") for i, x in enumerate(tag_data) if token_mask[i]],
			'str': [x for i, x in enumerate(texts) if token_mask[i]]
		}

		assert len(table) == len(texts)

		if self._stage == FlavorBuilder.Stage.PREFLIGHT:
			for k, xs in data.items():
				for x in xs:
					self._add_mappings(k, x)

		elif self._stage == FlavorBuilder.Stage.ADD:
			start = table["start"][token_mask]
			end = table["end"][token_mask]

			try:
				doc_group = self._documents_group.create_group(unique_id)
			except ValueError as err:
				raise RuntimeError(f"could not create group {unique_id} in {self._root}: {err}")

			doc_group.create_dataset(
				'token_mask',
				data=token_mask,
				dtype=np.bool)

			dset_tokens = doc_group.create_dataset(
				'span',
				(len(start), 2),
				dtype=np.uint32)

			dset_tokens[:, 0] = start
			dset_tokens[:, 1] = end - start

			n = start.shape[0]

			for k, v in data.items():
				mapping = self._mappings[k]
				dset = doc_group.create_dataset(
					k,
					(n,),
					dtype=np.min_scalar_type(len(mapping)))
				dset[:] = [mapping[x] for x in v]

		else:
			raise ValueError(f"unknown stage {self._stage}")

	def add(self, unique_id, doc):
		with doc.storage.tokens() as tokens:
			table = tokens.to_table()
			base_mask = self._token_to_token(table)

			with doc.storage.text() as text:
				self._add(unique_id, doc, text.get(), table, base_mask)


class EmbeddingCatalog:
	def __init__(self, path):
		self._conn = sqlite3.connect(path)

		self._conn.execute("""
			create table if not exists catalog (
				level varchar,
				key varchar,
				filename varchar unique,
				PRIMARY KEY (level, key)
			)""")

	def get_embeddings(self, level):
		cur = self._conn.cursor()
		try:
			cur.execute("""
				select key, filename from catalog where level=?
				""", (level,))
			rows = cur.fetchall()
		finally:
			cur.close()

		return rows

	def add_embedding(self, level, key):
		if not isinstance(key, str):
			key = json.dumps(key)

		cur = self._conn.cursor()
		try:
			cur.execute("""
				select filename from catalog where level=? and key=?
				""", (level, key,))
			unique_id = cur.fetchone()
		finally:
			cur.close()

		if unique_id is not None:
			return unique_id[0]

		unique_id = uuid.uuid1().hex

		with self._conn:
			self._conn.execute("""
				insert into catalog(level, key, filename) values (?, ?, ?)
				""", (level, key, unique_id))

		return unique_id


class Corpus:
	def __init__(self, path, mutable=False):
		path = Path(path)
		if not path.exists():
			path.mkdir()
		elif not path.is_dir():
			raise ValueError(f"expected directory path, got '{path}'")
		self._path = path

		self._documents_path = path / "documents"
		self._flavors_path = path / "flavors"
		self._flavors_path.mkdir(exist_ok=True)

		if not (path / "corpus.h5").exists():
			mutable = True

		self._corpus_h5 = h5py.File(path / "corpus.h5", "a" if mutable else "r")
		self._documents_group = self._corpus_h5.require_group("documents")
		#self._flavors_group = self._corpus_h5.require_group("flavors")

		self._corpus_sql = sqlite3.connect(path / "corpus.db")
		self._mutable = mutable

		with self._corpus_sql:
			self._corpus_sql.execute('''
				CREATE TABLE IF NOT EXISTS text(
					unique_id TEXT PRIMARY KEY,
					content TEXT,
					content_hash TEXT)''')
			self._corpus_sql.execute('''
				CREATE INDEX IF NOT EXISTS text_hash ON text(content_hash);
			''')

		data = threading.local()

		def create_catalog():
			return EmbeddingCatalog(path / "embeddings.db")

		def load_doc(unique_id):
			if 'catalog' in data.__dict__:
				embedding_catalog = data.catalog
			else:
				embedding_catalog = create_catalog()
				data.catalog = embedding_catalog

			p = self._documents_path / unique_id
			doc = Document.load_from_corpus(
				unique_id,
				p,
				self._corpus_sql,
				self._documents_group[unique_id],
				embedding_catalog)

			return unique_id, doc

		unique_ids = list(self._documents_group.keys())
		self._docs = {}

		self._doc_to_unique_id = {}
		self._unique_id_to_index = {}
		self._ordered_docs = []
		self._catalog = create_catalog()

		with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
			for unique_id, doc in tqdm(
				executor.map(load_doc, unique_ids),
				total=len(unique_ids),
				desc="Opening Corpus",
				disable=len(unique_ids) < 1):

				self._add_doc(unique_id, doc)

	def close(self):
		self._corpus_h5.close()
		self._conn.close()

	def iterdir(self):
		for k, v in self._documents_group.items():
			yield json.loads(v.attrs["metadata"])

	@property
	def embedding_catalog(self):
		return self._catalog

	def add_normalization(self, normalization):
		assert self._flavors_path.exists()
		path = self._flavors_path / f"{normalization.name}.h5"
		if path.exists():
			raise RuntimeError(f"Normalization {normalization.name} already exists")
		with h5py.File(path, 'w') as hf:
			FlavorBuilder.make(
				hf,
				normalization,
				self._docs)

	def del_normalization(self, normalization_name):
		p = self._flavors_path / f"{normalization_name}.h5"
		if p.exists() and p.is_file():
			p.unlink()

	@property
	def normalizations(self):
		names = []
		for p in self._flavors_path.iterdir():
			if p.suffix == ".h5":
				names.append(p.stem)
		return names

	@contextlib.contextmanager
	def flavor_cache(self, flavor_name):
		with h5py.File(self._flavors_path / f"{flavor_name}.h5", 'r') as hf:
			yield FlavorCache(hf)

	def get_unique_id(self, doc):
		if not isinstance(doc, Document):
			raise TypeError(f"{doc} is expected to be a Document")
		uid = self._doc_to_unique_id.get(id(doc))
		if uid is None:
			raise RuntimeError(f"unable to find {doc} in {self}")
		return uid

	def get_doc_index(self, doc):
		return self._unique_id_to_index[self._doc_to_unique_id[id(doc)]]

	def get_doc_path(self, doc):
		uid = self.get_unique_id(doc)
		return self._documents_path / uid

	def _add_doc(self, unique_id, doc):
		self._docs[unique_id] = doc
		self._doc_to_unique_id[id(doc)] = unique_id
		self._unique_id_to_index[unique_id] = len(self._ordered_docs)
		self._ordered_docs.append(doc)

	def add_doc(self, doc, ignore_dup=True):
		if not self._mutable:
			raise RuntimeError("corpus is not mutable")

		if ignore_dup:
			dup = doc.find_duplicates(self._corpus_sql)
			if len(dup) > 0:
				return False

		unique_id = str(uuid.uuid4())
		if unique_id in self._documents_group or unique_id in self._docs:
			raise ValueError("failed to create uuid for doc")

		try:
			doc.save_to_corpus(
				unique_id,
				self._documents_path / unique_id,
				self._corpus_sql,
				self._documents_group.create_group(unique_id),
				self._catalog)
		finally:
			self._corpus_h5.flush()

		self._add_doc(unique_id, doc)
		return True

	@property
	def path(self):
		return self._path

	@property
	def docs(self):
		return self._docs.values()

	def __len__(self):
		return len(self._docs)

	def __iter__(self):
		for doc in self._docs.values():
			yield doc

	def __getitem__(self, k):
		return self._ordered_docs[k]


class TemporaryCorpus(Corpus):
	@staticmethod
	def create_temp_path():
		temp_dir = Path.home() / ".vectorian" / "temp"
		temp_dir.mkdir(exist_ok=True, parents=True)
		return temp_dir

	@staticmethod
	def clear():
		shutil.rmtree(TemporaryCorpus.create_temp_path())

	def __init__(self):
		event_id = datetime.datetime.now().strftime(
			"%Y%m-%d%H-%M%S-") + str(uuid.uuid4())

		path = Path(tempfile.mkdtemp(
			prefix="vectorian_corpus_" + event_id,
			dir=TemporaryCorpus.create_temp_path()))
		assert len(list(path.iterdir())) == 0

		super().__init__(path)
