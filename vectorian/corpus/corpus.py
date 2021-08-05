import vectorian.core as core
import json
import concurrent.futures
import numpy as np
import h5py
import collections
import enum
import uuid

from vectorian.corpus.document import Document
from vectorian.importers import Importer
from pathlib import Path
from tqdm.autonotebook import tqdm
from collections import namedtuple
from slugify import slugify
from typing import Dict


class FlavorBuilder:
	class Stage(enum.Enum):
		PREFLIGHT = 0
		ADD = 1

	def __init__(self, root, normalizers):
		self._root = root

		self._normalizers = normalizers
		self._make_text = normalizers['token'].token_to_text
		self._token_to_token = normalizers['token'].token_to_token_many
		self._norm_text = normalizers['text'].to_callable()

		self._mappings = collections.defaultdict(dict)
		self._enums = {}

		self._stage = None

	@staticmethod
	def make(file, normalizers, docs):
		db = FlavorBuilder(file, normalizers)
		for stage in (FlavorBuilder.Stage.PREFLIGHT, FlavorBuilder.Stage.ADD):
			db.set_stage(stage)
			for unique_id, doc in docs.items():
				db.add(unique_id, doc)

	def _add_mappings(self, attr, value):
		m = self._mappings[attr]
		if value not in m:
			m[value] = len(m)

	def _make_enum(self, attr):
		mapping = self._mappings[attr]
		return h5py.enum_dtype(mapping, basetype=np.min_scalar_type(len(mapping)))

	def set_stage(self, stage: Stage):
		if stage == FlavorBuilder.Stage.ADD:
			for attr in self._mappings.keys():
				self._enums[attr] = self._make_enum(attr)
		self._stage = stage

	def _add(self, unique_id, doc, text, table, base_mask):
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

		if self._stage == FlavorBuilder.Stage.PREFLIGHT:
			for k, xs in data.items():
				for x in xs:
					self._add_mappings(k, x)

		elif self._stage == FlavorBuilder.Stage.ADD:
			start = table["start"][token_mask]
			end = table["end"][token_mask]

			try:
				doc_group = self._root.create_group(unique_id)
			except ValueError as err:
				raise RuntimeError(f"could not create group {unique_id} in {self._root}: {err}")

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

	def add(self, unique_id, doc):
		with doc.storage.tokens() as tokens:
			table = tokens.to_table()
			base_mask = self._token_to_token(table)

			with doc.storage.text() as text:
				self._add(unique_id, doc, text.get(), table, base_mask)


class Corpus:
	def __init__(self, path):
		path = Path(path)
		if not path.is_dir():
			raise ValueError(f"expected directory path, got '{path}'")
		self._path = path

		self._documents_path = path / "documents"
		self._documents_path.mkdir(exist_ok=True)

		self._corpus_h5 = h5py.File(path / "corpus.h5", "a")
		self._documents_group = self._corpus_h5.require_group("documents")
		self._flavors_group = self._corpus_h5.require_group("flavors")

		def load_doc(unique_id):
			p = self._documents_path / unique_id
			doc = Document.load_from_corpus(p, self._documents_group[unique_id])
			#doc.metadata["origin"] = p
			return unique_id, doc

		unique_ids = list(self._documents_group.keys())
		self._docs = {}

		self._doc_to_unique_id = {}
		self._unique_id_to_index = {}
		self._ordered_docs = []

		with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
			for unique_id, doc in executor.map(load_doc, unique_ids):
				self._add_doc(unique_id, doc)

	def close(self):
		self._corpus_h5.close()

	def add_flavor(self, flavor):
		FlavorBuilder.make(
			self._flavors_group.create_group(flavor.name),
			flavor.normalizers,
			self._docs)

	@property
	def flavors(self):
		return list(self._flavors_group.keys())

	def get_flavor(self, flavor):
		return self._flavors_group[flavor]

	def get_unique_id(self, doc):
		return self._doc_to_unique_id[id(doc)]

	def get_doc_index(self, doc):
		return self._unique_id_to_index[self._doc_to_unique_id[id(doc)]]

	def _add_doc(self, unique_id, doc):
		self._docs[unique_id] = doc
		self._doc_to_unique_id[id(doc)] = unique_id
		self._unique_id_to_index[unique_id] = len(self._ordered_docs)
		self._ordered_docs.append(doc)

	def add_doc(self, doc):
		unique_id = str(uuid.uuid4())
		if unique_id in self._documents_group or unique_id in self._docs:
			raise ValueError("failed to create uuid for doc")

		doc.save_to_corpus(
			self._documents_path / unique_id,
			self._documents_group.create_group(unique_id))

		self._add_doc(unique_id, doc)

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
