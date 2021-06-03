import numpy as np

from typing import List
from vectorian.sim.token import AbstractTokenSimilarity
from vectorian.sim.kernel import UnaryOperator, Kernel


class TokenSimilarityModifier(AbstractTokenSimilarity):
	@property
	def is_modifier(self):
		return True

	@property
	def name(self):
		raise NotImplementedError()

	@property
	def embeddings(self):
		raise NotImplementedError()


class UnaryTokenSimilarityModifier(TokenSimilarityModifier):
	def __init__(self, operand, operators: List[UnaryOperator]):
		self._operand = operand
		self._kernel = Kernel(operators)

	def __call__(self, operands, out):
		data = operands[0]

		out["similarity"][:] = data["similarity"]
		self._kernel(out["similarity"])

		for k in data.keys():
			if k != "similarity":
				out[k][:] = data[k]

	@property
	def operands(self):
		return [self._operand]

	@property
	def name(self):
		return self._kernel.name(self._operand.name)

	@property
	def embeddings(self):
		return [x.embedding for x in self._operand]


class MixedTokenSimilarity(TokenSimilarityModifier):
	def __init__(self, metrics, weights):
		self._metrics = metrics
		self._weights = weights

	@property
	def operands(self):
		return self._metrics

	def __call__(self, operands, out):
		for k in out.keys():
			data = [x[k] for x in operands]
			avg = np.average(data, axis=0, weights=self._weights)
			assert avg.shape == out[k].shape
			out[k][:] = avg

	@property
	def name(self):
		total = np.sum(self._weights)
		terms = []
		for m, w in zip(self._metrics, self._weights):
			terms.append(f'{w / total} * {m.name}')
		return f'({" + ".join(terms)})'

	@property
	def embeddings(self):
		return [x.embedding for x in self._metrics]


class ExtremumTokenSimilarity(TokenSimilarityModifier):
	def __init__(self, metrics):
		self._metrics = metrics

	@property
	def operands(self):
		return self._metrics

	def __call__(self, operands, out):
		sim = np.array([x["similarity"] for x in operands])

		sel = np.argmax(sim, axis=0)
		out["similarity"][:] = np.choose(
			sel.reshape(-1),
			sim.reshape(sim.shape[0], -1)).reshape(sim[0].shape)

		if "magnitudes" in out:
			weights = np.apply_along_axis(
				lambda arr: np.bincount(arr, minlength=sim.shape[0]), 1, sel)
			mag = np.array([x["magnitudes"] for x in operands])
			out["magnitudes"] = np.average(mag, axis=0, weights=weights.T)

	@property
	def name(self):
		return f'{self._name}({", ".join([x.name for x in self._metrics])})'

	@property
	def embeddings(self):
		return [x.embedding for x in self._metrics]


class MaximumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmax
	_name = 'maximum'


class MinimumTokenSimilarity(ExtremumTokenSimilarity):
	_operator = np.argmin
	_name = 'minimum'
