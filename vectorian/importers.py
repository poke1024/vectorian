import re
import logging
import datetime
import numpy as np
import download

from tqdm.autonotebook import tqdm
from pathlib import Path
from collections import namedtuple

from vectorian.embeddings import ContextualEmbedding, Vectors, ProxyVectorsRef


def normalize_dashes(s):
	s = re.sub(r"(\w)\-(\s)", r"\1 -\2", s)
	s = re.sub(r"(\s)\-(\w)", r"\1- \2", s)
	return s


def to_min_dtype(array):
	# note that we do not check for min and assume it's 0 or -1
	max = np.max(array)
	for dtype in (np.int8, np.int16, np.int32, np.int64):
		if max <= np.iinfo(dtype).max:
			return array.astype(dtype)
	raise ValueError(f"failed to map value {max} to numpy")


def compile_doc_spans(tokens):
	doc_spans = {
		'start': np.array([0], dtype=np.int32),
		'end': np.array([len(tokens)], dtype=np.int32)
	}

	return doc_spans


def compile_spans(spans, tokens, loc_ax):
	n = len(spans['start'])

	new_spans = {
		'start': np.empty(n, dtype=np.int32),
		'end': np.empty(n, dtype=np.int32)
	}

	loc = np.array(spans['loc'])
	for i, k in enumerate(loc_ax):
		assert k not in new_spans
		new_spans[k] = to_min_dtype(loc[:, i])

	token_i = 0

	new_start = new_spans['start']
	new_end = new_spans['end']

	for i, (start, end) in enumerate(zip(spans['start'], spans['end'])):
		if start > tokens[token_i]['start']:
			raise RuntimeError(
				f"unexpected span start {start} vs. {tokens[token_i]['start']}")

		token_j = token_i + 1
		while token_j < len(tokens):
			if tokens[token_j]['start'] >= end:
				break
			token_j += 1

		new_start[i] = token_i
		new_end[i] = token_j

		token_i = token_j

	return new_spans


def make_tokens_dict(tokens):
	dtypes = {
		'id': 'int',
		'start': 'int',
		'end': 'int',
		'head': 'int',
		'tag': 'enum',
		'pos': 'enum',
		'dep': 'enum'
	}

	keys = set.union(*[set(t.keys()) for t in tokens])
	res = dict()
	for k in keys:
		data = [t.get(k) for t in tokens]
		dtype = dtypes.get(k, 'str')
		if dtype == 'int':
			res[k] = {
				'dtype': 'int',
				'data': to_min_dtype(np.array(data, dtype=np.int64))
			}
		else:
			res[k] = {
				'dtype': dtype,
				'data': data
			}
	return res


Metadata = namedtuple(
	"Metadata", ["version", "unique_id", "origin", "author", "title", "speakers"])


class Importer:
	_t = {
		'normalize-dashes': normalize_dashes
	}

	def __init__(self, nlp, embeddings=None, preprocessors=None, batch_size=1):
		if embeddings is None:
			embeddings = []
		if preprocessors is None:
			preprocessors = ["normalize-dashes"]

		for embedding in embeddings:
			if not isinstance(embedding, ContextualEmbedding):
				raise TypeError(f"expected ContextualEmbedding, got {embedding}")

		self._nlp = nlp
		self._embeddings = embeddings
		self._batch_size = batch_size
		# batch_size == 1 needed for https://github.com/explosion/spaCy/issues/3607
		self._transforms = preprocessors

	def _preprocess_text(self, text):
		for x in self._transforms:
			if callable(x):
				text = x(text)
			else:
				text = Importer._t[x](text)
		return text

	def _make_doc(self, md, partitions, loc_ax, locations, show_progress=True):
		pipe = self._nlp.pipe(
			partitions,
			batch_size=self._batch_size,
			disable=['ner', 'lemmatizer'])  # check nlp.pipe_names

		contextual_vectors = dict((e.name, []) for e in self._embeddings)
		texts = []
		tokens = []

		sents = {
			'start': [],
			'end': [],
			'loc': []
		}

		text_len = 0

		for location, doc in tqdm(
				zip(locations, pipe),
				total=len(locations),
				desc=f'Importing {md.origin}',
				disable=not show_progress):

			doc_json = doc.to_json()

			partition_tokens = doc_json['tokens']
			for token in partition_tokens:
				token['start'] += text_len
				token['end'] += text_len
			tokens.extend(partition_tokens)

			for sent in doc_json['sents']:
				sents['start'].append(text_len + sent['start'])
				sents['end'].append(text_len + sent['end'])
				sents['loc'].append(location)

			texts.append(doc_json['text'])
			text_len += len(doc_json['text'])

			for e in self._embeddings:
				contextual_vectors[e.name].append(e.encode(doc))

		spans = {
			'sentence': compile_spans(sents, tokens, loc_ax),
			'document': compile_doc_spans(tokens)
		}

		json = {
			'metadata': md._asdict(),
			'loc_ax': loc_ax
		}

		from vectorian.corpus.document import Document, InternalMemoryDocumentStorage

		emb_by_name = dict((e.name, e) for e in self._embeddings)

		def transformed(k, v):
			v = Vectors(np.vstack(v))

			embedding = emb_by_name[k]
			if embedding.transform:
				v = embedding.transform.apply(v)

			return ProxyVectorsRef(v)

		contextual_embeddings = dict(
			(k, transformed(k, v)) for k, v in contextual_vectors.items())

		return Document(
			InternalMemoryDocumentStorage(json, ''.join(texts), make_tokens_dict(tokens), spans),
			contextual_embeddings)


class TextImporter(Importer):
	# an importer for plain text files that does not assume any structure.

	def __call__(self, path, unique_id=None, author="Anonymous", title=None):
		path = Path(path)

		if title is None:
			title = path.stem

		if unique_id is None:
			unique_id = f"{author}/{title}"

		with open(path, "r") as f:
			text = self._preprocess_text(f.read())

		locations = []

		paragraph_sep = "\n\n"
		paragraphs = text.split(paragraph_sep)
		paragraphs = [x + paragraph_sep for x in paragraphs[:-1]] + paragraphs[-1:]

		for j, p in enumerate(paragraphs):
			locations.append((j,))

		md = Metadata(
			version="1.0",
			unique_id=unique_id,
			origin=str(path),
			author=author,
			title=title,
			speakers={})

		return self._make_doc(
			md, paragraphs, ['paragraph'], locations)


class NovelImporter(Importer):
	# a generic importer for novel-like texts.

	_chapters = re.compile(
		r"\n\n\n\W*chapter\s+(\d+)[^\n]*\n\n", re.IGNORECASE)

	def __call__(self, path, unique_id=None, author="Anonymous", title=None):
		path = Path(path)

		if title is None:
			title = path.stem

		if unique_id is None:
			unique_id = f"{author}/{title}"

		with open(path, "r") as f:
			text = self._preprocess_text(f.read())

		chapter_breaks = []
		expected_chapter = 1
		book = 1
		for m in NovelImporter._chapters.finditer(text):
			actual_chapter = int(m.group(1))

			if expected_chapter != actual_chapter:
				if book == 1 and expected_chapter == 2 and actual_chapter == 1:
					# we might have received "chapter 1"
					# as part of the table of contents.
					chapter_breaks = []
					expected_chapter = 1
				elif actual_chapter == 1:
					book += 1
					expected_chapter = 1
				else:
					logging.warning("bad chapter. wanted %d, got: %s" % (
						expected_chapter, m.group(0).strip()))
					chapter_breaks = []
					break

			chapter_breaks.append((book, actual_chapter, m.start(0)))
			expected_chapter += 1

		chapters = dict()
		if chapter_breaks:
			chapter_breaks.append((book, actual_chapter + 1, len(text)))
			for ((book, chapter, s), (_, _, e)) in zip(chapter_breaks, chapter_breaks[1:]):
				chapters[(book, chapter)] = text[s:e]

			first_break = chapter_breaks[0][2]
			if first_break > 0:
				chapters[(-1, -1)] = text[:first_break]
		else:
			chapters[(-1, -1)] = text

		paragraphs = []
		locations = []
		ignore_book = book <= 1

		for book, chapter in sorted(chapters.keys()):
			chapter_text = chapters[(book, chapter)]
			if ignore_book:
				book = -1

			chapter_sep = "\n\n"
			chapter_paragraphs = chapter_text.split(chapter_sep)
			chapter_paragraphs = [x + chapter_sep for x in chapter_paragraphs[:-1]] + chapter_paragraphs[-1:]
			paragraphs.extend(chapter_paragraphs)

			for j, p in enumerate(chapter_paragraphs):
				locations.append((book + 1, chapter + 1, j + 1))

		md = Metadata(
			version="1.0",
			unique_id=unique_id,
			origin=str(path),
			author=author,
			title=title,
			speakers={})

		return self._make_doc(
			md, paragraphs, ['book', 'chapter', 'paragraph'], locations)


class BodleianImporter(Importer):
	# import for the TEI format files used in the Bodleian library.
	pass


class PlayShakespeareImporter(Importer):
	# an importer for the PlayShakespeare.com Shakespeare XMLs available at
	# https://github.com/severdia/PlayShakespeare.com-XML

	def download(self, name, category="playshakespeare_editions"):
		cache_path = Path.home() / ".vectorian" / "texts" / "playshakespeare" / category
		cache_path.mkdir(exist_ok=True, parents=True)
		xml_path = cache_path / name

		if not xml_path.exists():
			url = f"https://raw.githubusercontent.com/severdia/PlayShakespeare.com-XML/master/{category}/{name}"
			download.download(url, xml_path, progressbar=True)

		return self(xml_path)

	def __call__(self, path):
		import xml.etree.ElementTree as ET
		from collections import defaultdict

		path = Path(path)
		tree = ET.parse(path)
		root = tree.getroot()
		speakers = defaultdict(int)
		full_speaker_names = dict()

		for persname in root.findall(".//persname"):
			full_speaker_names[persname.attrib["short"]] = persname.text

		locations = []
		texts = []

		scenes = list(root.findall(".//scene"))

		for scene_index, scene in enumerate(scenes):
			actnum = int(scene.attrib["actnum"])
			scenenum = int(scene.attrib["num"])

			for speech in scene.findall(".//speech"):
				speaker = speech.find("speaker")

				speaker_no = speakers.get(speaker.text)
				if speaker_no is None:
					speaker_no = len(speakers) + 1
					speakers[speaker.text] = speaker_no

				line_no = None
				lines = []
				for line in speech.findall("line"):
					if line.text:
						if line_no is None:
							line_no = int(line.attrib["globalnumber"])
						lines.append(line.text)

				if lines:
					locations.append((actnum, scenenum, speaker_no, line_no))
					texts.append(normalize_dashes(" ".join(lines)))

		unique_name = f"{root.attrib['variant']}-{root.attrib['unique']}"

		md = Metadata(
			version="1.0",
			unique_id="PlayShakespeare.com/William Shakespeare/play/" + unique_name,
			origin=str(path),
			author="William Shakespeare",
			title=root.find(".//title").text,
			speakers={v: full_speaker_names.get(k, k) for k, v in speakers.items()})

		return self._make_doc(
			md, texts, ['act', 'scene', 'speaker', 'line'], locations)


class StringImporter(Importer):
	# a generic importer for short text strings for ad-hoc experiments.

	def __call__(self, s, unique_id=None, author="", title="", show_progress=False):
		if unique_id is None:
			unique_id = str(datetime.datetime.now())

		locations = [
			[]
		]

		md = Metadata(
			version="1.0",
			unique_id=unique_id,
			origin="<string>",
			author=author,
			title=title,
			speakers={})

		return self._make_doc(
			md, [self._preprocess_text(s)], [], locations, show_progress=show_progress)
