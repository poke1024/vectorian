import vectorian.core as core
import numpy as np
import scipy.signal
import scipy.ndimage.filters

from vectorian.corpus.document import PreparedDocument
from vectorian.session import Partition


class Filter:
	def __call__(self, x):
		raise NotImplementedError()


class ConvFilter(Filter):
	def __init__(self, pulse):
		self._pulse = pulse / np.sum(pulse)

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


class GaussFilter(ConvFilter):
	def __init__(self, width, fc=1):
		t = np.linspace(-1, 1, width, endpoint=True)
		_, e = scipy.signal.gausspulse(t, fc=fc, retenv=True)
		super().__init__(e)


class MaxFilter(Filter):
	def __init__(self, width):
		self._size = width

	def __call__(self, x):
		return scipy.ndimage.filters.maximum_filter(
			x, size=self._size)


class Signal:
	_filters = {
		'gauss': GaussFilter,
		'max': MaxFilter
	}

	def __call__(self, doc: PreparedDocument, partition: Partition):
		raise NotImplementedError()

	def smoothed(self, width, method="max"):
		return SmoothedSignal(self, Signal._filters[method](width))


class SmoothedSignal(Signal):
	def __init__(self, base, filter_):
		self._base = base
		self._filter = filter_

	def __call__(self, doc: PreparedDocument, partition: Partition):
		return self._filter(self._base(doc, partition))


class _FastKeywordSignal(Signal):
	def __init__(self, *keywords, max_count=1):
		self._keywords = keywords
		self._max_count = max_count

	def __call__(self, doc: PreparedDocument, partition: Partition):
		counts = doc.compiled.count_keywords(
			partition.to_args(),
			self._keywords)

		counts = np.minimum(counts, self._max_count)

		return counts.astype(np.float32) / self._max_count


class CustomSignal(Signal):
	def spans_to_signal(self, spans):
		raise NotImplementedError()

	def __call__(self, doc: PreparedDocument, partition: Partition):
		spans = list(doc.spans(partition))
		signal = self.spans_to_signal(spans)
		assert np.max(signal) <= 1
		assert np.min(signal) >= 0
		return signal


class KeywordSignal(CustomSignal):
	def __init__(self, *keywords, max_count=1, same=None):
		self._keywords = set(keywords)
		self._max_count = max_count
		self._same = same

	def _check(self, x):
		if self._same is None:
			return x in self._keywords
		else:
			for y in self._keywords:
				if self._same(x, y):
					return True
			return False

	def spans_to_signal(self, spans):
		w = np.zeros((len(spans),), dtype=np.float32)
		for i, span in enumerate(spans):
			n = 0
			for token in span:
				if self._check(token.text):
					n += 1
			w[i] = n

		w = np.minimum(w, self._max_count)

		return w / self._max_count


class Saliency:
	def __init__(self, strength=0.5):
		self._f = []
		self._w = []
		self._strength = strength
		if self._strength < 0 or self._strength > 1:
			raise ValueError(f"strength has illegal value {strength}")

	def add_signal(self, signal: Signal, weight:float = 1.0):
		def compute(doc: PreparedDocument, partition: Partition):
			return signal(doc, partition).astype(np.float32)

		self._f.append(compute)
		self._w.append(weight)

	def compile(self, doc: PreparedDocument, partition: Partition):
		w_sum = np.sum(self._w)
		normal_w = np.array(self._w) / w_sum

		signals = [np.ones((doc.n_spans(partition),), dtype=np.float32)]
		signals.extend([f(doc, partition) for f in self._f])

		if len(signals) == 1:
			w = [1]
		else:
			w = [1 - self._strength] + (normal_w * self._strength).tolist()

		return core.Booster(
			np.average(signals, axis=0, weights=w))
