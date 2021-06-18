import numpy as np
import io

from typing import Dict


class GapCost:
	@property
	def is_affine(self):
		return False

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

	@property
	def is_affine(self):
		return self._cost == 0

	def to_scalar(self):
		assert self._cost == 0
		return 0

	def to_description(self):
		return f'ConstantGapCost({self._cost})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c.fill(self._cost)
		c[0] = 0
		return c


class GotohGapCost(GapCost):
	"""
	\( w_k = u k + v \)
	"""
	pass


class AffineGapCost(GapCost):
	"""
	\( w_k = u k \)
	"""

	def __init__(self, u):
		self._u = u

	@property
	def is_affine(self):
		return True

	def to_scalar(self):
		return self._u

	def to_description(self):
		return f'AffineGapCost({self.u})'

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		x = 0
		for i in range(1, n):
			c[i] = x
			x += self._u
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


class GlobalAlignment(AlignmentStrategy):
	"""
	Models global alignments as originally described by Needleman and Wunsch (1970).

	To align two sequences of length n, the runtime complexity is \\( O(n^2) \\) if
	the gap cost is affine, and \\( O(n^3) \\) otherwise.

	As the runtime of Needleman and Wunsch's original algorithm is always cubic and does not
	include gap costs, the implementation follows Sankoff (1972) and adopts linear (affine)
	gap costs as described by Kruskal (1983).

	The case with affine gap costs is sometimes referred to as Needleman-Wunsch algorithm.


	Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable
	to the search for similarities in the amino acid sequence of two proteins.
	Journal of Molecular Biology, 48(3), 443–453. https://doi.org/10.1016/0022-2836(70)90057-4

	Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
	the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

	Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
	String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045

	Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275
	"""

	def __init__(self, gap: Dict[str, GapCost]):
		self._gap = gap
		if not all(k in ("s", "t") for k in gap.keys()):
			raise ValueError(gap)

	def to_description(self, partition):
		return {
			'GlobalAlignment': {
				'gap': self._gap
			}
		}

	def to_args(self, partition):
		if all(x.is_affine for x in self._gap.values()):
			return {
				'algorithm': 'alignment/global/affine',
				'gap': dict((k, v.to_scalar()) for k, v in self._gap.items())
			}
		else:
			len = partition.max_len() + 1
			return {
				'algorithm': 'alignment/global/general',
				'gap': dict((k, v.costs) for k, v in self._gap.items())
			}


#class SemiGlobalAlignment(AlignmentStrategy):
#	def __init__(self, gap: Dict[str, GapCost]):
#		pass


class LocalAlignment(AlignmentStrategy):
	"""
	Models local alignments as described by Smith, Waterman and Beyer (1976).

	The dynamic programming approach used is similar to that used in global alignments,
	but adds an additional "zero" case and a modified traceback to generate local alignments
	(for details see Aluru).

	To align two sequences of length n, the runtime complexity is \\( O(n^2) \\) if
	the gap cost is affine, and \\( O(n^3) \\) otherwise.

	The case dealing with general gap costs is sometimes referred to as Waterman-Smith-Beyer
	algorithm, whereas the simpler case assuming affine gap costs is sometimes called the
	Smith-Waterman algorithm.

	For the affine gap cost case, the implementation follows Sankoff (1972) and Kruskal (1983).


	Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
	the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

	Waterman, M. S., Smith, T. F., & Beyer, W. A. (1976). Some biological sequence metrics.
	Advances in Mathematics, 20(3), 367–387. https://doi.org/10.1016/0001-8708(76)90202-4

	Smith, T. F., & Waterman, M. S. (1981). Identification of common molecular subsequences.
	Journal of Molecular Biology, 147(1), 195–197. https://doi.org/10.1016/0022-2836(81)90087-5

	Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
	String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045

	Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275
	"""

	def __init__(self, gap: Dict[str, GapCost], zero: float = 0):
		self._gap = gap
		self._zero = zero
		if not all(k in ("s", "t") for k in gap.keys()):
			raise ValueError(gap)

	def to_description(self, partition):
		return {
			'LocalAlignment': {
				'gap': self._gap,
				'zero': self._zero
			}
		}

	def to_args(self, partition):
		if all(x.is_affine for x in self._gap.values()):
			return {
				'algorithm': 'alignment/local/affine',
				'gap': dict((k, v.to_scalar()) for k, v in self._gap.items()),
				'zero': self._zero
			}
		else:
			len = partition.max_len() + 1
			return {
				'algorithm': 'alignment/local/general',
				'gap': dict((k, v.costs) for k, v in self._gap.items()),
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
		"""
		Args:
			symmetric: indicates whether to also solve the symmetric RWMD problem,
				which gives a tighter lower bound of WMD, but doubles the computation
				cost
		"""

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
