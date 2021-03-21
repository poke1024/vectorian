import re
import logging
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import namedtuple

from vectorian.embeddings import ContextualEmbedding, InMemoryVectorsRef


def normalize_dashes(s):
	s = re.sub(r"(\w)\-(\s)", r"\1 -\2", s)
	s = re.sub(r"(\s)\-(\w)", r"\1- \2", s)
	return s


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

	def _make_doc(self, md, partitions, loc_keys, locations):
		pipe = self._nlp.pipe(
			partitions,
			batch_size=self._batch_size,
			disable=['ner', 'lemmatizer'])  # check nlp.pipe_names

		contextual_vectors = dict((e.name, []) for e in self._embeddings)

		json_partitions = []
		for location, doc in tqdm(zip(locations, pipe), total=len(locations), desc=f'Importing {md.origin}'):
			doc_json = doc.to_json()
			doc_json['loc'] = location
			json_partitions.append(doc_json)

			for e in self._embeddings:
				contextual_vectors[e.name].append(e.encode(doc))

		json = {
			'metadata': md._asdict(),
			'partitions': json_partitions,
			'loc_keys': loc_keys
		}

		from vectorian.corpus import Document

		contextual_embeddings = dict(
			(k, InMemoryVectorsRef(np.vstack(v))) for k, v in contextual_vectors.items())

		return Document(json, contextual_embeddings)


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

'''

         <div n="CI" type="chapter">
            <head>VARIATION UNDER
DOMESTICATION</head>

'''


class ShakespeareImporter(Importer):
	# an importer for the PlayShakespeare.com Shakespeare XMLs available at
	# https://github.com/severdia/PlayShakespeare.com-XML

	def __call__(self, path, unique_id=None, author="Anonymous", title=None):
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

		md = Metadata(
			version="1.0",
			unique_id="William Shakespeare/" + root.find(".//title").text,
			origin=str(path),
			author="William Shakespeare",
			title=root.find(".//title").text,
			speakers={v: full_speaker_names.get(k, k) for k, v in speakers.items()})

		return self._make_doc(
			md, texts, ['act', 'scene', 'speaker', 'line'], locations)


class StringImporter(Importer):
	# a generic importer for short text strings for ad-hoc experiments.

	def __call__(self, s, unique_id=None, author="", title=""):
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

		return self._make_doc(md, [self._preprocess_text(s)], [], locations)
