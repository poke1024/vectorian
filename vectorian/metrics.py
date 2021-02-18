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


class SqrtCosineMetric(Metric):
	def __init__(self, embedding):
		pass


class MixedMetric(Metric):
	def __init__(self, a: Metric, b: Metric, t: float):
		self._a = a
		self._b = b
		self._t = t

	def to_args(self):
		return {
			'name': 'mix',
			'embedding': None,
			'metric': 'mix',
			'options': {
				'a': self._a.to_args(),
				'b': self._b.to_args(),
				't': self._t
			}
		}

