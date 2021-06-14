import numpy as np
import io


class GapCost:
	def costs(self, n):
		raise NotImplementedError

	def _plot(self, ax, n):
		from matplotlib.ticker import MaxNLocator
		c = self.costs(n)
		ax.plot(c)
		ax.set_xlabel('gap length')
		ax.set_ylabel('cost')
		ax.set_ylim(-0.1, 1)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.grid()

	def plot(self, n):
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(1, 1, figsize=(12, 3))
		self._plot(ax, n)
		fig.tight_layout()
		fig.show()

	def plot_to_image(self, fig, ax, n, format='png'):
		self._plot(ax, n)
		buf = io.BytesIO()
		fig.tight_layout()
		fig.savefig(buf, format=format)
		buf.seek(0)
		data = buf.getvalue()
		buf.close()
		return data

	def _ipython_display_(self):
		# see https://ipython.readthedocs.io/en/stable/config/integrating.html
		self.plot(50)


class ConstantGapCost(GapCost):
	"""
	Models a constant gap cost \( C(t) = c_0 \) (i.e. independent of
	gap length).
	"""

	def __init__(self, cost):
		self._cost = cost

	def to_description(self):
		return f'ConstantGapCost({self._cost})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c.fill(self._cost)
		c[0] = 0
		return c


class LinearGapCost(GapCost):
	"""
	Models a gap cost C of the form \( C(t) = c_0 + t \\dot d \),
	with \( c_0 \) being an initial cost and \( d \) being an affine
	factor.
	"""

	def __init__(self, step, start=None):
		self._step = step
		self._start = step if start is None else start

	def to_scalar(self):
		assert self._start == self._step
		return self._step

	def to_description(self):
		return f'LinearGapCost({self._step}, {self._start})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		x = self._start
		for i in range(1, n):
			c[i] = x
			x += self._step
		return c


class ExponentialGapCost(GapCost):
	def __init__(self, cutoff):
		self._cutoff = cutoff

	def to_description(self):
		return f'ExponentialGapCost({self._cutoff})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		if self._cutoff > 0:
			for i in range(n):
				c[i] = 1 - (2 ** -(i / self._cutoff))
		else:
			c.fill(1)
			c[0] = 0
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
		return c


class SpanFlowStrategy:
	def to_description(self, partition):
		raise NotImplementedError()

	def to_args(self, partition):
		raise NotImplementedError()


class AlignmentStrategy(SpanFlowStrategy):
	"""
	A strategy to align two sequences - i.e. matching one sequence against the
	other sequences through a number of insertions and deletions while keeping
	the order of both sequences.
	"""

	pass


class TransportStrategy(SpanFlowStrategy):
	"""
	A strategy to match two sequences by posing a transport problem in a network.
	The order of tokens is not inherently important in such formulations.
	"""

	pass


class NeedlemanWunsch(AlignmentStrategy):
	"""
	Models the global alignment algorithm by Needleman and Wunsch (1970).

	The runtime of Needleman and Wunsch's original algorithm is cubic and does not include
	gap costs, the implementation therefore follows Sankoff (1972) and adopts linear (affine)
	gap costs as described by Kruskal (1983).

	To align two sequences of length m and n, runtime complexity is \\( O(m n) \\).

	Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable
	to the search for similarities in the amino acid sequence of two proteins.
	Journal of Molecular Biology, 48(3), 443–453. https://doi.org/10.1016/0022-2836(70)90057-4

	Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
	the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

	Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
	String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045
	"""

	def __init__(self, gap: float = 0, gap_mask="st"):
		"""
		Args:
			gap (float): the cost of inserting or deleting a token
			gap_mask (str): whether to apply `gap` to s, t, or both
		"""


		self._gap = gap
		self._gap_mask = gap_mask

	def to_description(self, partition):
		return {
			'NeedlemanWunsch': {
				'gap': self._gap
			}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'needleman-wunsch',
			'gap': self._gap,
			'gap_mask': self._gap_mask
		}


class SmithWaterman(AlignmentStrategy):
	"""
	Models the local alignment algorithm by Smith and Waterman (for affine gap costs).

	This is basically the global alignment algorithm by Needleman and Wunsch
	(1970), adapted to produce a local alignment by using an additional "zero" case and
	a slightly modified traceback (for details, see Aluru, for example).

	The runtime of Needleman and Wunsch's original algorithm is cubic and does not include
	gap costs, the implementation therefore follows Sankoff (1972) and adopts linear (affine)
	gap costs as described by Kruskal (1983).

	To align two sequences of length m and n, runtime complexity is \\( O(m n) \\).

	Smith, T. F., & Waterman, M. S. (1981). Identification of common molecular subsequences.
	Journal of Molecular Biology, 147(1), 195–197. https://doi.org/10.1016/0022-2836(81)90087-5

	Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
	the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

	Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
	String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045

	Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275
	"""

	def __init__(self, gap: float = 0, gap_mask="st", zero: float = 0.5):
		"""
		Args
			gap (float): the cost of inserting or deleting a token
			gap_mask (str): whether to apply `gap` to s, t, or both
			zero (float): a measure of when two elements (tokens) are not similar enough - this
				is a crucial parameter in constructing a local alignment.
		"""

		self._gap = gap
		self._gap_mask = gap_mask
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
			'gap_mask': self._gap_mask,
			'zero': self._zero
		}


class WatermanSmithBeyer(AlignmentStrategy):
	"""
	Models the local alignment algorithm by Waterman, Smith and Beyer.

	This does not impose any restrictions on the structure of the gap cost, e.g. gap cost
	need to be affine (i.e. a linear function of gap length as with Waterman-Smith).

	To align two sequences of length n, runtime complexity is \\( O(n^3) \\).

	.. note::
	   Some literature refers to this algorithm as Smith-Waterman.

	Waterman, M. S., Smith, T. F., & Beyer, W. A. (1976). Some biological sequence metrics.
	Advances in Mathematics, 20(3), 367–387. https://doi.org/10.1016/0001-8708(76)90202-4
	"""

	def __init__(self, gap: GapCost = None, gap_mask="st", zero: float = 0.5):
		if gap is None:
			gap = ConstantGapCost(0)
		self._gap = gap
		self._gap_mask = gap_mask
		self._zero = zero

	def to_description(self, partition):
		return {
			'WatermanSmithBeyer': {
				'gap': self._gap.to_description(),
				'zero': self._zero
			}
		}

	def to_args(self, partition):
		costs = self._gap.costs(partition.max_len())

		return {
			'algorithm': 'waterman-smith-beyer',
			'gap': np.clip(costs, 0, 1),
			'gap_mask': self._gap_mask,
			'zero': self._zero
		}


class WordMoversDistance(TransportStrategy):
	"""
	Implements various variants of the Word Mover's Distance. The original full
	WMD described by Kusner et al. can be instantiated through `wmd("nbow")`. The
	original RWMD described by Kusner et al. can be instantiated through `rwmd("nbow")`.

	Kusner, M. J., Sun, Y., Kolkin, N. I., & Weinberger, K. Q. (2015). From word
	embeddings to document distances. Proceedings of the 32nd International Conference
	on International Conference on Machine Learning - Volume 37, 957–966.

	Atasu, K., Parnell, T., Dünner, C., Sifalakis, M., Pozidis, H., Vasileiadis, V.,
	Vlachos, M., Berrospi, C., & Labbi, A. (2017). Linear-Complexity Relaxed Word Mover’s
	Distance with GPU Acceleration. ArXiv:1711.07227 [Cs]. http://arxiv.org/abs/1711.07227
	"""

	@staticmethod
	def wmd(variant='bow', **kwargs):
		"""
		Create a variant of WMD. To compute the WMD for two documents/sentences, the runtime
		is super cubic in the number of different tokens involved.
		"""

		kwargs['builtin'] = f"wmd/{variant}"
		if variant == 'bow':
			return WordMoversDistance(False, False, False, True, **kwargs)
		elif variant == 'nbow':
			return WordMoversDistance(False, False, False, False, **kwargs)
		else:
			raise ValueError(variant)

	@staticmethod
	def rwmd(variant, **kwargs):
		"""
		Create a variant of Relaxed WMD (RWMD).

		In contrast to WMD, RWMD solves a much simpler problem and is therefore
		much faster to compute. To compute the RWMD for two documents/sentences,
		the runtime is \( O(n) \) if \( n \) is the number of different tokens
		involved.
		"""

		kwargs['builtin'] = f"rwmd/{variant}"
		if variant == 'nbow':
			return WordMoversDistance(True, True, True, True, **kwargs)
		elif variant == 'nbow/distributed':  # i.e. jablonsky
			return WordMoversDistance(True, False, True, True, **kwargs)
		elif variant == 'bow/fast':  # non-symmetric, injective
			return WordMoversDistance(True, True, False, False, **kwargs)
		else:
			raise ValueError(variant)

	def __init__(
		self, relaxed=True, injective=True, symmetric=False, normalize_bow=False,
		extra_mass_penalty=-1, builtin=None):

		self._options = {
			'relaxed': relaxed,
			'injective': injective,
			'normalize_bow': normalize_bow,
			'symmetric': symmetric,
			'extra_mass_penalty': extra_mass_penalty
		}

		self._builtin_name = builtin

	@property
	def builtin_name(self):
		return self._builtin_name

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
			'normalize_bow': self._options['normalize_bow'],
			'extra_mass_penalty': self._options['extra_mass_penalty']
		}


class WordRotatorsDistance(TransportStrategy):
	"""
	Implements the Word Rotators Distance by Yokoi et al.

	Yokoi, S., Takahashi, R., Akama, R., Suzuki, J., & Inui, K. (2020).
	Word Rotator’s Distance. Proceedings of the 2020 Conference on
	Empirical Methods in Natural Language Processing (EMNLP), 2944–2960.
	https://doi.org/10.18653/v1/2020.emnlp-main.236
	"""

	def __init__(self, normalize_magnitudes=True, extra_mass_penalty=-1):
		self._normalize_magnitudes = normalize_magnitudes
		self._extra_mass_penalty = extra_mass_penalty

	def to_description(self, partition):
		return {
			'WordRotatorsDistance': {
				'normalize_magnitudes': self._normalize_magnitudes,
				'extra_mass_penalty': self._extra_mass_penalty
			}
		}

	def to_args(self, partition):
		return {
			'algorithm': 'word-rotators-distance',
			'normalize_magnitudes': self._normalize_magnitudes,
			'extra_mass_penalty': self._extra_mass_penalty
		}
