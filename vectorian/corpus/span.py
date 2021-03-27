import numpy as np
import collections


class SpansTable:
	_types = {
		'token_at': np.uint32,
		'n_tokens': np.uint16
	}

	def __init__(self, loc_keys):
		self._loc = collections.defaultdict(list)
		self._loc_keys = loc_keys
		self._max_n_tokens = np.iinfo(SpansTable._types['n_tokens']).max

	def extend(self, location, n_tokens):
		if n_tokens < 1:
			return

		loc = self._loc

		for k, v in zip(self._loc_keys, location):
			loc[k].append(v)

		if loc['token_at']:
			token_at = loc['token_at'][-1] + loc['n_tokens'][-1]
		else:
			token_at = 0

		if n_tokens > self._max_n_tokens:
			raise RuntimeError(f'n_tokens = {n_tokens} > {self._max_n_tokens}')

		loc['n_tokens'].append(n_tokens)
		loc['token_at'].append(token_at)

	def to_dict(self):
		data = dict()
		for k, v in self._loc.items():
			dtype = SpansTable._types.get(k)
			if dtype is None:
				series = np.array(v)
			else:
				series = np.array(v, dtype=dtype)
			data[k] = series
		return data
