import vectorian.core as core
import json
import concurrent.futures
import numpy as np
import h5py
import collections
import enum

from vectorian.corpus.document import Document
from vectorian.importers import Importer
from pathlib import Path
from tqdm.autonotebook import tqdm
from collections import namedtuple
from typing import Dict


class CorpusDB:
	class Stage(enum.Enum):
		PREFLIGHT = 0
		ADD = 1

	def __init__(self, path, normalizers):
		self._file = h5py.File(path, "w")

		self._normalizers = normalizers
		self._make_text = normalizers['token'].token_to_text
		self._token_to_token = normalizers['token'].token_to_token_many
		self._norm_text = normalizers['text'].to_callable()

		self._mappings = collections.defaultdict(dict)
		self._enums = {}

		self._stage = None

	def close(self):
		self._file.close()

	@staticmethod
	def make(path, normalizers, docs):
		db = CorpusDB(path, normalizers)
		try:
			for stage in (CorpusDB.Stage.PREFLIGHT, CorpusDB.Stage.ADD):
				db.set_stage(stage)
				for doc in docs:
					db.add(doc)
		finally:
			db.close()

	def _add_mappings(self, attr, value):
		m = self._mappings[attr]
		if value not in m:
			m[value] = len(m)

	def _make_enum(self, attr):
		mapping = self._mappings[attr]
		return h5py.enum_dtype(mapping, basetype=np.min_scalar_type(len(mapping)))

	def set_stage(self, stage: Stage):
		if stage == CorpusDB.Stage.ADD:
			for attr in self._mappings.keys():
				self._enums[attr] = self._make_enum(attr)
		self._stage = stage

	def _add(self, doc, text, table, base_mask):
		texts = list(map(
			self._norm_text,
			self._normalizers['token'].token_to_text_many(text, table)))
		token_mask = np.logical_and(
			base_mask,
			np.array([x and len(x.strip()) > 0 for x in texts], dtype=np.bool))

		data = {
			'pos': [(x or "") for i, x in enumerate(table["pos"]) if token_mask[i]],
			'tag': [(x or "") for i, x in enumerate(table["tag"]) if token_mask[i]],
			'str': [x for i, x in enumerate(texts) if token_mask[i]]
		}

		assert len(table) == len(texts)

		if self._stage == CorpusDB.Stage.PREFLIGHT:
			for k, xs in data.items():
				for x in xs:
					self._add_mappings(k, x)

		elif self._stage == CorpusDB.Stage.ADD:
			start = table["start"][token_mask]
			end = table["end"][token_mask]

			doc_group = self._file.create_group(doc.unique_id)

			doc_group.attrs['origin'] = str(doc.metadata['origin'])
			doc_group.attrs['token_mask'] = token_mask

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
					dtype=self._enums[k])
				dset[:] = [mapping[x] for x in v]

		else:
			raise ValueError(f"unknown stage {self._stage}")

	def add(self, doc):
		with doc.storage.tokens() as tokens:
			table = tokens.to_table()
			base_mask = self._token_to_token(table)

			with doc.storage.text() as text:
				self._add(doc, text.get(), table, base_mask)


class Corpus:
	def __init__(self, docs=None):
		if docs is None:
			docs = []
		self._docs = list(docs)
		self._ids = set([doc.unique_id for doc in self._docs])

	def add(self, doc):
		assert doc.unique_id not in self._ids
		self._docs.append(doc)
		self._ids.add(doc.unique_id)

	@staticmethod
	def _create_corpus_json(path):
		path = Path(path)
		names = []
		for p in path.iterdir():
			if p.suffix == ".document":
				names.append(p.name)
		with open(path / "corpus.json", "w") as f:
			f.write(json.dumps({
				'documents': sorted(names)
			}))

	@staticmethod
	def load(path):
		path = Path(path)

		if not (path / "corpus.json").exists():
			Corpus._create_corpus_json(path)
		with open(path / "corpus.json", "r") as f:
			names = json.loads(f.read())["documents"]

		def load_doc(name):
			p = path / name
			doc = Document.load(p)
			doc.metadata["origin"] = p
			return doc

		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
			docs = executor.map(load_doc, names)

		return Corpus(docs)

	def save(self, path):
		names = set()
		for doc in self._docs:
			name = doc.unique_id
			if name not in names:
				names.add(name)
			else:
				raise ValueError(f"non-unique document name '{name}'")

		path = Path(path)
		path.mkdir(exist_ok=True)
		for doc in self._docs:
			doc_path = path / (doc.caching_name + ".document")
			doc_path.mkdir(exist_ok=False)
			doc.save(doc_path)
		Corpus._create_corpus_json(path)

	def __len__(self):
		return len(self._docs)

	def __iter__(self):
		for doc in self._docs:
			yield doc

	def __getitem__(self, k):
		return self._docs[k]

	def save_compact(self, path, normalizers: Dict):
		CorpusDB.make(path, normalizers, self._docs)


class LazyCorpus:
	def __init__(self, name):
		self._src = []
		self._path = Path.home() / ".vectorian" / "corpus" / name
		self._path.mkdir(exist_ok=True, parents=True)

	def add(self, path, importer: Importer, unique_id=None):
		assert isinstance(importer, Importer)

		Source = namedtuple("Source", [
			"path",
			"importer",
			"unique_id"
		])

		path = Path(path)
		if path.is_dir():
			for p in path.iterdir():
				self.add(p, importer=importer)
		else:
			if unique_id is None:
				unique_id = path.stem
			self._src.append(Source(
				path=path,
				importer=importer,
				unique_id=unique_id
			))

	def _compile(self):
		for src in tqdm(self._src):
			p = self._path / (src.caching_name + ".json")
			if p.exists():
				continue
			doc = src.importer(src.path)
			doc.save(p)

	def __iter__(self):
		self._compile()

		for p in sorted(self._path.iterdir()):
			doc = Document.load(p)
			doc["origin"] = p
			yield doc
