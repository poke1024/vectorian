import vectorian.core as core
import numpy as np
import io

from typing import Dict, Union
from pyalign.gaps import *


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

	def _make_args(self, locality, gaps):
		return {
			'algorithm': 'pyalign',
			'options': {
				'locality': locality,
				'gap_cost': gaps
			}
		}


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

	def __init__(self, gap: Union[GapCost, Dict[str, GapCost]]):
		self._gap = gap
		if isinstance(gap, Dict):
			if not all(k in ("s", "t") for k in gap.keys()):
				raise ValueError(gap)

	@property
	def gap(self):
		return self._gap

	def to_description(self, partition):
		return {
			'GlobalAlignment': {
				'gap': self._gap
			}
		}

	def to_args(self, partition):
		return self._make_args(core.pyalign.Locality.GLOBAL, self._gap)


class SemiGlobalAlignment(AlignmentStrategy):
	"""
	Models semiglobal (also called "end gaps free" or "free-shift") alignments.

	In this variant, insertions and deletions at the start and at the end of
	the sequences are not weighted as penalties.

	Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275
	"""

	def __init__(self, gap: Union[GapCost, Dict[str, GapCost]]):
		self._gap = gap
		if isinstance(gap, Dict):
			if not all(k in ("s", "t") for k in gap.keys()):
				raise ValueError(gap)

	@property
	def gap(self):
		return self._gap

	def to_description(self, partition):
		return {
			'SemiGlobalAlignment': {
				'gap': self._gap
			}
		}

	def to_args(self, partition):
		return self._make_args(core.pyalign.Locality.SEMIGLOBAL, self._gap)


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

	def __init__(self, gap: Union[GapCost, Dict[str, GapCost]]):
		self._gap = gap
		if isinstance(gap, Dict):
			if not all(k in ("s", "t") for k in gap.keys()):
				raise ValueError(gap)

	@property
	def gap(self):
		return self._gap

	def to_description(self, partition):
		return {
			'LocalAlignment': {
				'gap': self._gap,
				'zero': self._zero
			}
		}

	def to_args(self, partition):
		return self._make_args(core.pyalign.Locality.LOCAL, self._gap)


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
		Create a variant of WMD. To compute the WMD for two documents/sentences, the
		runtime is super cubic in the number of different tokens involved.
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
			relaxed: indicates whether to solve a RWMD or a full WMD.

			normalize_bow: indicates whether to use nbow or (unnormalized) bow as
				representation of the bag of words.

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
