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
	pass


class IsolatedMetric(SentenceMetric):
	def __init__(self, word_metric):
		assert isinstance(word_metric, WordMetric)
		self._word_metric = word_metric

	def to_args(self):
		return {
			'metric': 'isolated',
			'word_metric': self._word_metric.to_args()
		}


class TagWeightedMetric(SentenceMetric):
	def __init__(self, word_metric, tag_weights, pos_mismatch_penalty, similarity_threshold):
		assert isinstance(word_metric, WordMetric)
		self._word_metric = word_metric

	def to_args(self):
		return {
			'metric': 'tag_weighted',
			'word_metric': self._word_metric.to_args()
		}
