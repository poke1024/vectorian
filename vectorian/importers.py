import re
import logging
import datetime

from tqdm import tqdm
from pathlib import Path
from collections import namedtuple

from vectorian.embeddings import ContextualEmbedding


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
			batch_size=self._batch_size)

		json_partitions = []
		for location, doc in tqdm(zip(locations, pipe), total=len(locations), desc='Importing: ' + md.title):
			doc_json = doc.to_json()
			doc_json['loc'] = location
			json_partitions.append(doc_json)

			for embedding in self._embeddings:
				v = embedding.encode(doc)
				print('??', v.shape, len(doc))

		json = {
			"version": md.version,
			'unique_id': md.unique_id,
			'origin': str(md.origin),
			'author': md.author,
			'title': md.title,
			'speakers': md.speakers,
			'partitions': json_partitions,
			'loc_keys': loc_keys
		}

		from vectorian.corpus import Document

		return Document(json)


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
			origin=path,
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
					logging.warn("bad chapter. wanted %d, got: %s" % (
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
			origin=path,
			author=author,
			title=title,
			speakers={})

		return self._make_doc(
			md, paragraphs, ['book', 'chapter', 'paragraph'], locations)


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
