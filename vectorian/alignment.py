import numpy as np
import numbers


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

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		for i in range(n):
			c[i] = 1 - (2 ** -(i / self._cutoff))
		c = np.clip(c, 0, 1)
		return c


class CustomGapCost:
	def __init__(self, costs_fn):
		self._costs_fn = costs_fn


class WatermanSmithBeyer:
	def __init__(self, gap=np.inf, zero=0.5):
		self._gap = gap
		self._zero = zero

	def to_args(self, session):
		return {
			'algorithm': 'wsb',
			'gap': self._gap.costs(session.max_sentence_len),
			'zero': self._zero
		}
