from vectorian.alignment import WatermanSmithBeyer
from vectorian.index import BruteForceIndex


class WordMetric:
	pass


class CosineMetric(WordMetric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-cosine",
			'embedding': self._embedding.name,
			'metric': 'cosine',
			'options': {}
		}


class ZhuCosineMetric(WordMetric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-zhu-cosine",
			'embedding': self._embedding.name,
			'metric': 'zhu-cosine',
			'options': {}
		}


class SohangirCosineMetric(WordMetric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-sohangir-cosine",
			'embedding': self._embedding.name,
			'metric': 'sohangir-cosine',
			'options': {}
		}


class PNormMetric(WordMetric):
	def __init__(self, embedding, p=2, scale=1):
		self._embedding = embedding
		self._p = p
		self._scale = scale

	def to_args(self):
		return {
			'name': self._embedding.name + "-p-norm",
			'embedding': self._embedding.name,
			'metric': 'p-norm',
			'options': {
				'p': self._p,
				'scale': self._scale
			}
		}


class LerpMetric(WordMetric):
	def __init__(self, a: WordMetric, b: WordMetric, t: float):
		self._a = a
		self._b = b
		self._t = t

	def to_args(self):
		return {
			'name': 'lerp',
			'embedding': None,
			'metric': 'lerp',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args(),
				't': self._t
			}
		}


class MinMetric(WordMetric):
	def __init__(self, a: WordMetric, b: WordMetric):
		self._a = a
		self._b = b

	def to_args(self):
		return {
			'name': 'min',
			'embedding': None,
			'metric': 'min',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args()
			}
		}


class MaxMetric(WordMetric):
	def __init__(self, a: WordMetric, b: WordMetric):
		self._a = a
		self._b = b

	def to_args(self):
		return {
			'name': 'max',
			'embedding': None,
			'metric': 'max',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args()
			}
		}


class SentenceMetric:
	def create_index(self, session):
		raise NotImplementedError()

	def to_args(self, session):
		raise NotImplementedError()


class AlignmentSentenceMetric(SentenceMetric):
	def __init__(self, word_metric, alignment=None):
		assert isinstance(word_metric, WordMetric)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._word_metric = word_metric
		self._alignment = alignment

	def create_index(self, session):
		return BruteForceIndex(session, self)

	def to_args(self, session):
		return {
			'metric': 'alignment-isolated',
			'word_metric': self._word_metric.to_args(),
			'alignment': self._alignment.to_args(session)
		}


class TagWeightedSentenceMetric(SentenceMetric):
	def __init__(self, word_metric, alignment, tag_weights, pos_mismatch_penalty, similarity_threshold):
		assert isinstance(word_metric, WordMetric)

		if alignment is None:
			alignment = WatermanSmithBeyer()

		self._word_metric = word_metric
		self._alignment = alignment

	def create_index(self, session):
		return BruteForceIndex(session, self)

	def to_args(self, session):
		return {
			'metric': 'alignment-tag-weighted',
			'word_metric': self._word_metric.to_args(),
			'alignment': self._alignment.to_args(session)
		}
