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


def rewrite(rules):
	def f(t):
		t_new = t.copy()
		for k, v in rules.items():
			t_new[k] = v.get(t[k], t[k])
		return t_new
	return f


def to_mapper(self, mode):
	elements = [x for x in self._mappings if getattr(x, mode)]

