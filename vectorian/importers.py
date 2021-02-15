import re
import logging

from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from vectorian.corpus.document import Document


def normalize_dashes(s):
	s = re.sub(r"(\w)\-(\s)", r"\1 -\2", s)
	s = re.sub(r"(\s)\-(\w)", r"\1- \2", s)
	return s


Metadata = namedtuple(
	"Metadata", ["version", "unique_id", "author", "title", "speakers"])


class Importer:
	def __init__(self, nlp, batch_size=1):
		self._nlp = nlp
		self._batch_size = batch_size
		# batch_size == 1 needed for https://github.com/explosion/spaCy/issues/3607

	def _make_doc(self, md, partitions, locations):
		pipe = self._nlp.pipe(
			partitions,
			batch_size=self._batch_size)

		json_partitions = []
		for location, doc in tqdm(zip(locations, pipe), total=len(locations)):
			doc_json = doc.to_json()
			doc_json['loc'] = {
				'bk': location[0],  	# book
				'ch': location[1], 		# chapter
				'sp': location[2], 		# speaker
				'l': location[3]		# line
			}
			json_partitions.append(doc_json)

		json = {
			"version": md.version,
			'id': md.unique_id,
			'author': md.author,
			'title': md.title,
			'speakers': md.speakers,
			'partitions': json_partitions
		}

		return Document(json)


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
			text = normalize_dashes(f.read())

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
				locations.append((book, chapter, -1, j))

		md = Metadata(
			version="1.0",
			unique_id=unique_id,
			author=author,
			title=title,
			speakers={})

		return self._make_doc(md, paragraphs, locations)
