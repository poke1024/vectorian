import vectorian.core as core
import numpy as np
import scipy.signal

from vectorian.corpus.document import PreparedDocument
from vectorian.session import Partition


class Convolve:
	def __call__(self, x):
		raise NotImplementedError()


class GaussPulse(Convolve):
	def __init__(self, width, fc):
		t = np.linspace(-1, 1, width, endpoint=True)
		_, e = scipy.signal.gausspulse(t, fc=fc, retenv=True)
		self._pulse = e / np.sum(e)

	def __call__(self, x):
		if self._pulse.shape[0] <= x.shape[0]:
			y = np.convolve(
				x, self._pulse, mode='same')
			return y
		else:
			return x

	def _ipython_display_(self):
		import matplotlib.pyplot
		matplotlib.pyplot.plot(self._pulse)


class Signal:
	def __call__(self, doc: PreparedDocument, partition: Partition):
		raise NotImplementedError()


class KeywordSignal(Signal):
	def __init__(self, *keywords, max_count=1):
		self._keywords = keywords
		self._max_count = max_count

	def __call__(self, doc: PreparedDocument, partition: Partition):
		counts = doc.compiled.count_keywords(
			partition.to_args(),
			self._keywords)

		counts = np.minimum(counts, self._max_count)

		return counts.astype(np.float32) / self._max_count


class Booster:
	def __init__(self, strength=0.5):
		self._f = []
		self._w = []
		self._strength = strength
		if self._strength < 0 or self._strength > 1:
			raise ValueError(f"strength has illegal value {strength}")

	def add(self, signal: Signal, convolve:Convolve = None, weight:float = 1.0):
		def compute(doc: PreparedDocument, partition: Partition):
			w = signal(doc, partition)
			if convolve:
				w = convolve(w)
			return w.astype(np.float32)

		self._f.append(compute)
		self._w.append(weight)

	def compile(self, doc: PreparedDocument, partition: Partition):
		if not np.isclose(np.sum(self._w), 1):
			raise ValueError("weights must sum up to 1")

		signals = [np.ones((doc.n_spans(partition),), dtype=np.float32)]
		signals.extend([f(doc, partition) for f in self._f])

		if len(signals) == 1:
			w = [1]
		else:
			w = [1 - self._strength] + (np.array(self._w) * self._strength).tolist()

		return core.Booster(
			np.average(signals, axis=0, weights=w))
