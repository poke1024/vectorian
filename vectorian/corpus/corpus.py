import vectorian.core as core

from vectorian.corpus.document import Document
from vectorian.importers import Importer
from pathlib import Path
from tqdm.autonotebook import tqdm
from collections import namedtuple


class Corpus:
	def __init__(self, docs=None):
		if docs is None:
			docs = []
		self._docs = docs
		self._ids = set([doc.unique_id for doc in docs])

	def add(self, doc):
		assert doc.unique_id not in self._ids
		self._docs.append(doc)
		self._ids.add(doc.unique_id)

	@staticmethod
	def load(path):
		path = Path(path)
		docs = []
		for p in sorted(path.iterdir()):
			if p.suffix == ".txt":
				doc = Document.load(p)
				doc.metadata["origin"] = p
				docs.append(doc)
		return Corpus(docs)

	def save(self, path):
		path = Path(path)
		path.mkdir(exist_ok=True)
		for doc in self._docs:
			doc.save(path / (doc.caching_name + ".json"))

	def __iter__(self):
		for doc in self._docs:
			yield doc


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
