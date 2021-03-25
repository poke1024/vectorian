import re


def chain(callables):
	def call(arg):
		for f in callables:
			arg = f(arg)
			if arg is None:
				break
		return arg

	return call


class CachableCallable:
	def __init__(self, name, callable_):
		self._name = name
		self._callable = callable_

	@property
	def name(self):
		return self._name

	def __call__(self, *args, **kwargs):
		return self._callable(*args, **kwargs)

	def unpack(self):
		return self._callable

	@staticmethod
	def chain(callables):
		name = "_".join([x.name for x in callables])
		unpacked = [x.unpack() for x in callables]
		return CachableCallable(name, chain(unpacked))


def lowercase():
	return CachableCallable('lowercase', lambda s: s.lower())


def erase(mode=None):
	if mode == "W":
		pattern = re.compile(r"\W")
		return CachableCallable('eraseW', lambda s: pattern.sub("", s))
	else:
		raise ValueError(mode)


def strip():
	return CachableCallable('strip', lambda s: s.strip())


def alpha():
	return CachableCallable('alpha', lambda s: s if s.isalpha() else None)


class RemoveTokens:
	def __init__(self, tokens):
		self._tokens = set(tokens)

	def __call__(self, s):
		return None if s in self._tokens else s


def stop_words(words):
	return CachableCallable('stop', RemoveTokens(words))


def rewrite(rules):
	def f(t):
		t_new = t.copy()
		for k, v in rules.items():
			t_new[k] = v.get(t[k], t[k])
		return t_new
	return f


def filter_punct(t):
	return None if t["pos"] == "PUNCT" else t


def default_token_mappings():
	# Vectorian's default token mappings. You might want
	# to adjust this by adding lowercase mapping and/or
	# other pos tag mappings.

	return {
		"tokenizer": [
			erase("W"),  # remove any non-word characters
			alpha()  # ignore any tokens that are empty
		],
		"tagger": [
			filter_punct,  # ignore punctuation tokens
			rewrite({
				# rewrite PROPN as NOUN to fix accuracy

				'pos': {
					'PROPN': 'NOUN'
				},
				'tag': {
					'NNP': 'NN',
					'NNPS': 'NNS',
				}
			})
		]
	}
