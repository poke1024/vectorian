import numpy as np

from typing import List


class UnaryOperator:
	def kernel(self, data: np.ndarray):
		raise NotImplementedError()

	def name(self, operand):
		raise NotImplementedError()


class RadialBasis(UnaryOperator):
	def __init__(self, gamma: float):
		self._gamma = gamma

	def kernel(self, data: np.ndarray):
		data[:, :] = np.exp(-self._gamma * np.power(data, 2))

	def name(self, operand):
		return f'radialbasis({operand}, {self._gamma})'


class DistanceToSimilarity(UnaryOperator):
	def kernel(self, data: np.ndarray):
		data[:, :] = np.maximum(0, 1 - data)

	def name(self, operand):
		return f'(1 - {operand})'


class Bias(UnaryOperator):
	def __init__(self, bias: float):
		self._bias = bias

	def kernel(self, data: np.ndarray):
		data += self._bias

	def name(self, operand):
		return f'({operand} + {self._bias})'


class Scale(UnaryOperator):
	def __init__(self, scale: float):
		self._scale = scale

	def kernel(self, data: np.ndarray):
		data *= self._scale

	def name(self, operand):
		return f'({operand} * {self._scale})'


class Power(UnaryOperator):
	def __init__(self, exp: float):
		self._exp = exp

	def kernel(self, data: np.ndarray):
		data[:, :] = np.power(np.maximum(data, 0), self._exp)

	def name(self, operand):
		return f'({operand} ** {self._exp})'


class Threshold(UnaryOperator):
	def __init__(self, threshold: float):
		self._threshold = threshold

	def kernel(self, data: np.ndarray):
		x = np.maximum(0, data - self._threshold)
		x[x > 0] += self._threshold
		data[:, :] = x

	def name(self, operand):
		return f'threshold({operand}, {self._threshold})'


class Kernel:
	def __init__(self, operators: List[UnaryOperator]):
		self._operators = operators

		# might JIT in the future.
		def chain(data):
			for x in operators:
				x.kernel(data)

		self._kernel = chain

	def __call__(self, data: np.ndarray):
		self._kernel(data)

	def name(self, operand):
		name = operand
		for x in self._operators:
			name = x.name(name)
		return name
