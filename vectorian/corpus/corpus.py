import vectorian.core as core
import json
import concurrent.futures

from vectorian.corpus.document import Document
from vectorian.importers import Importer
from pathlib import Path
from tqdm.autonotebook import tqdm
from collections import namedtuple


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
			if p.suffix == ".txt":
				names.append(p.name)
		with open(path / "corpus.json", "w") as f:
			f.write(json.dumps({
				'docs': sorted(names)
			}))

	@staticmethod
	def load(path):
		path = Path(path)

		if not (path / "corpus.json").exists():
			Corpus._create_corpus_json(path)
		with open(path / "corpus.json", "r") as f:
			names = json.loads(f.read())["docs"]

		def load_doc(name):
			p = path / name
			doc = Document.load(p)
			doc.metadata["origin"] = p
			return doc

		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
			docs = executor.map(load_doc, names)

		return Corpus(docs)

	def save(self, path):
		path = Path(path)
		path.mkdir(exist_ok=True)
		for doc in self._docs:
			doc.save(path / (doc.caching_name + ".json"))
		Corpus._create_corpus_json(path)

	def __len__(self):
		return len(self._docs)

	def __iter__(self):
		for doc in self._docs:
			yield doc

	def __getitem__(self, k):
		return self._docs[k]


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
