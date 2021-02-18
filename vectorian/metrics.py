class Metric:
	pass


class CosineMetric(Metric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-cosine",
			'embedding': self._embedding.name,
			'metric': 'cosine',
			'options': {}
		}


class ZhuCosineMetric(Metric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-zhu-cosine",
			'embedding': self._embedding.name,
			'metric': 'zhu-cosine',
			'options': {}
		}


class SohangirCosineMetric(Metric):
	def __init__(self, embedding):
		self._embedding = embedding

	def to_args(self):
		return {
			'name': self._embedding.name + "-sohangir-cosine",
			'embedding': self._embedding.name,
			'metric': 'sohangir-cosine',
			'options': {}
		}


class PNormMetric(Metric):
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


class LerpMetric(Metric):
	def __init__(self, a: Metric, b: Metric, t: float):
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


class MinMetric(Metric):
	def __init__(self, a: Metric, b: Metric):
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


class MaxMetric(Metric):
	def __init__(self, a: Metric, b: Metric):
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
