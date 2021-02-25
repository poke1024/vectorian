import vectorian.core as core
import logging
import roman

from cached_property import cached_property
from functools import lru_cache

from vectorian.render import Renderer
from vectorian.metrics import CosineMetric, WordSimilarityMetric, AlignmentSentenceMetric, SentenceSimilarityMetric
from vectorian.embeddings import StaticEmbedding


def get_location_desc(metadata, location):
	book = location["book"]
	chapter = location["chapter"]
	speaker = location["speaker"]
	paragraph = location["paragraph"]

	if speaker > 0:  # we have an act-scene-speakers structure.
		speaker = metadata["speakers"].get(str(speaker), "")
		if book >= 0:
			act = roman.toRoman(book)
			scene = chapter
			return speaker, "%s.%d, line %d" % (act, scene, paragraph)
		else:
			return speaker, "line %d" % paragraph
	elif chapter > 0:  # book, chapter and paragraphs
		if book < 0:  # do we have a book?
			return "", "Chapter %d, par. %d" % (chapter, paragraph)
		else:
			return "", "Book %d, Chapter %d, par. %d" % (
				book, chapter, paragraph)
	else:
		return "", "par. %d" % paragraph


def result_set_to_json(items):
	matches = []
	for i, m in enumerate(items):
		regions = []
		doc = m.document
		sentence = doc.sentence(m.sentence)

		try:
			for r in m.regions:
				s = r.s.decode('utf-8', errors='ignore')
				if r.matched:
					t = r.t.decode('utf-8', errors='ignore')
					regions.append(dict(
						s=s,
						t=t,
						similarity=r.similarity,
						weight=r.weight,
						pos_s=r.pos_s.decode('utf-8', errors='ignore'),
						pos_t=r.pos_t.decode('utf-8', errors='ignore'),
						metric=r.metric.decode('utf-8', errors='ignore')))
				else:
					regions.append(dict(s=s, mismatch_penalty=r.mismatch_penalty))

			metadata = m.document.metadata
			speaker, loc_desc = get_location_desc(metadata, sentence)

			matches.append(dict(
				debug=dict(document=m.document.id, sentence=m.sentence),
				score=m.score,
				metric=m.metric,
				location=dict(
					speaker=speaker,
					author=metadata["author"],
					title=metadata["title"],
					location=loc_desc
				),
				regions=regions,
				omitted=m.omitted))
		except UnicodeDecodeError:
			logging.exception("unicode conversion issue")

	return matches


class Result:
	def __init__(self, matches, duration):
		self._matches = matches
		self._duration = duration

	def __iter__(self):
		return self._matches

	def __getitem__(self, i):
		return self._matches[i]

	@lru_cache(1)
	def to_json(self):
		return result_set_to_json(self._matches)

	def limit_to(self, n):
		return type(self)(self._matches[:n])

	@property
	def duration(self):
		return self._duration


class Collection:
	def __init__(self, vocab, corpus, filter_):
		self._vocab = vocab
		self._docs = []
		for doc in corpus:
			self._docs.append(
				doc.to_core(len(self._docs), self._vocab, filter_)
			)
			doc.free_up_memory()

	@property
	def documents(self):
		return self._docs

	@cached_property
	def max_sentence_len(self):
		return max([doc.max_sentence_len for doc in self._docs])


class DefaultImportFilter:
	_pos = {
		'PROPN': 'NOUN'
	}

	_tag = {
		'NNP': 'NN',
		'NNPS': 'NNS',
	}

	def __call__(self, t):
		if t["pos"] == "PUNCT":
			return False

		# spaCy is very generous with labeling things as PROPN,
		# which really breaks pos_mimatch_penalty often. we re-
		# classify PROPN as NOUN.
		t_new = t.copy()
		t_new["pos"] = self._pos.get(t["pos"], t["pos"])
		t_new["tag"] = self._tag.get(t["tag"], t["tag"])
		return t_new


class Session:
	def __init__(self, docs, embeddings, import_filter=DefaultImportFilter()):
		self._vocab = core.Vocabulary()
		self._default_metrics = []
		for embedding in embeddings:
			if not isinstance(embedding, StaticEmbedding):
				raise TypeError(f"expected StaticEmbedding, got {embedding}")
			self._vocab.add_embedding(embedding.to_core())
			self._default_metrics.append(
				AlignmentSentenceMetric(
					WordSimilarityMetric(
						embedding, CosineMetric())))
		self._collection = Collection(self._vocab, docs, import_filter)

	@property
	def documents(self):
		return self._collection.documents

	@property
	def vocab(self):
		return self._vocab

	@property
	def max_sentence_len(self):
		return self._collection.max_sentence_len

	@property
	def result_class(self):
		return Result

	def make_index(self, metric=None):
		if metric is None:
			metric = self._default_metrics[0]
		assert isinstance(metric, SentenceSimilarityMetric)
		return metric.create_index(self)

	def run_query(self, find, query):
		return Result, find(query)


class LabResult(Result):
	def __init__(self, matches, duration, annotate=None):
		super().__init__(matches, duration)
		self._annotate = annotate

	def _render(self, r):
		# see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
		for match in self.to_json():
			r.add_match(match)
		return r.to_html()

	def annotate(self, tags=True, metric=True, **kwargs):
		return LabResult(
			self._matches,
			self._duration,
			annotate=dict(tags=tags, metric=metric, **kwargs))

	def _repr_html_(self):
		return self._render(Renderer(annotate=self._annotate))


class LabSession(Session):
	def run_query(self, find, query):
		import ipywidgets as widgets
		from IPython.display import display

		progress = widgets.FloatProgress(
			value=0, min=0, max=1, description="",
			layout=widgets.Layout(width="100%"))

		display(progress)

		def update_progress(t):
			progress.value = progress.max * t

		try:
			result = find(
				query,
				progress=update_progress)
		finally:
			progress.close()

		return LabResult, result
