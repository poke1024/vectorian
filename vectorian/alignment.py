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


class CustomGapCost(GapCost):
	def __init__(self, costs_fn):
		self._costs_fn = costs_fn

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		for i in range(1, n):
			c[i] = self._costs_fn(i)
		c = np.clip(c, 0, 1)
		return c


class WatermanSmithBeyer:
	def __init__(self, gap: GapCost = None, zero: float = 0.5):
		if gap is None:
			gap = ConstantGapCost(0)
		self._gap = gap
		self._zero = zero

	def to_args(self, session, options):
		slices = options['slices']
		return {
			'algorithm': 'wsb',
			'gap': self._gap.costs(session.max_len(
				slices['level'], slices['window_size'])),
			'zero': self._zero
		}


class WordMoversDistance:
	_variants = {
		'kusner': {
			'multiplicity': 'n:n',
			'repr': 'nbow',
			'symmetric': True
		},
		'relaxed-kusner': {
			'multiplicity': '1:1',
			'repr': 'nbow',
			'symmetric': True
		},
		'relaxed-vectorian': {
			'multiplicity': '1:1',
			'repr': 'bow',
			'symmetric': False,
		},
		'relaxed-jablonsky': {
			'multiplicity': '1:n',
			'repr': 'nbow',
			'symmetric': True,
		}
	}

	def __init__(self, variant=None, options=None):
		if options is None:
			if variant is None:
				variant = 'relaxed-vectorian'
			self._options = WordMoversDistance._variants.get(variant)
			if self._options is None:
				raise ValueError(f"unsupported WMD variant {variant}")
		else:
			assert variant is None
			self._options = options

	def to_args(self, session, options):
		multiplicity = self._options['multiplicity']
		representation = self._options['repr']
		symmetric = self._options['symmetric']

		if multiplicity == "1:1":
			one_target = True
		elif multiplicity == "1:n":
			one_target = False
		elif multiplicity == "n:n":
			raise NotImplementedError("full wmd is not yet implemented")
		else:
			raise ValueError(f"unsupported multiplicity {multiplicity}")

		if representation == "bow":
			normalize_bow = False
		elif representation == "nbow":
			normalize_bow = True
		else:
			raise ValueError("repr must be either 'bow' or 'nbow'")

		return {
			'algorithm': 'rwmd',
			'one_target': one_target,
			'normalize_bow': normalize_bow,
			'symmetric': symmetric
		}
