import numpy as np


class GapCost:
	def costs(self, n):
		raise NotImplementedError

	def plot(self, n):
		import matplotlib.pyplot as plt
		c = self.costs(n)
		plt.figure(figsize=(12, 3))
		plt.plot(c)
		plt.xlabel('gap length')
		plt.ylabel('cost')
		plt.tight_layout()
		plt.grid()
		plt.show()

	def _ipython_display_(self):
		# see https://ipython.readthedocs.io/en/stable/config/integrating.html
		self.plot(50)


class ConstantGapCost(GapCost):
	def __init__(self, cost):
		self._cost = cost

	def to_description(self):
		return f'ConstantGapCost({self._cost})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c.fill(self._cost)
		c[0] = 0
		c = np.clip(c, 0, 1)
		return c


class LinearGapCost(GapCost):
	def __init__(self, step, start=None):
		self._step = step
		self._start = step if start is None else start

	def to_description(self):
		return f'LinearGapCost({self._step}, {self._start})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		x = self._start
		for i in range(1, n):
			c[i] = x
			x += self._step
		c = np.clip(c, 0, 1)
		return c


class ExponentialGapCost(GapCost):
	def __init__(self, cutoff):
		self._cutoff = cutoff

	def to_description(self):
		return f'ExponentialGapCost({self._cutoff})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		for i in range(n):
			c[i] = 1 - (2 ** -(i / self._cutoff))
		c = np.clip(c, 0, 1)
		return c


class CustomGapCost(GapCost):
	def __init__(self, costs_fn):
		self._costs_fn = costs_fn

	def to_description(self):
		return f'CustomGapCost({self._costs_fn})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		for i in range(1, n):
			c[i] = self._costs_fn(i)
		c = np.clip(c, 0, 1)
		return c


class AlignmentAlgorithm:
	def to_description(self, partition):
		raise NotImplementedError()

	def to_args(self, partition):
		raise NotImplementedError()


class NeedlemanWunsch(AlignmentAlgorithm):
	def __init__(self, gap: float = 0):
		self._gap = gap

	def to_description(self, partition):
		return {
			'NeedlemanWunsch': {
				'gap': self._gap
			}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'needleman-wunsch',
			'gap': self._gap
		}


class SmithWaterman(AlignmentAlgorithm):
	def __init__(self, gap: float = 0, zero: float = 0.5):
		self._gap = gap
		self._zero = zero

	def to_description(self, partition):
		return {
			'WatermanSmithBeyer': {
				'gap': self._gap,
				'zero': self._zero
			}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'smith-waterman',
			'gap': self._gap,
			'zero': self._zero
		}


class WatermanSmithBeyer(AlignmentAlgorithm):
	def __init__(self, gap: GapCost = None, zero: float = 0.5):
		if gap is None:
			gap = ConstantGapCost(0)
		self._gap = gap
		self._zero = zero

	def to_description(self, partition):
		return {
			'WatermanSmithBeyer': {
				'gap': self._gap.to_description(),
				'zero': self._zero
			}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'waterman-smith-beyer',
			'gap': self._gap.costs(partition.max_len()),
			'zero': self._zero
		}


class WordMoversDistance(AlignmentAlgorithm):
	@staticmethod
	def wmd(variant='kusner'):
		if variant == 'kusner':
			return WordMoversDistance(False, False, False, True)
		elif variant == 'vectorian':
			return WordMoversDistance(False, False, False, False)
		else:
			raise ValueError(variant)

	@staticmethod
	def rwmd(variant):
		if variant == 'kusner':
			return WordMoversDistance(True, True, True, True)
		elif variant == 'jablonsky':
			return WordMoversDistance(True, False, True, True)
		elif variant == 'vectorian':
			return WordMoversDistance(True, True, False, False)
		else:
			raise ValueError(variant)

	def __init__(self, relaxed=True, injective=True, symmetric=False, normalize_bow=False, extra_mass_penalty=-1):
		self._options = {
			'relaxed': relaxed,
			'injective': injective,
			'normalize_bow': normalize_bow,
			'symmetric': symmetric,
			'extra_mass_penalty': extra_mass_penalty
		}

	def to_description(self, partition):
		return {
			'WordMoversDistance': self._options
		}

	def to_args(self, partition):
		return {
			'algorithm': 'word-movers-distance',
			'relaxed': self._options['relaxed'],
			'injective': self._options['injective'],
			'symmetric': self._options['symmetric'],
			'normalize_bow': self._options['normalize_bow']
		}


class WordRotatorsDistance(AlignmentAlgorithm):
	def __init__(self):
		pass

	def to_description(self, partition):
		return {
			'WordRotatorsDistance': {}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'word-rotators-distance'
		}
