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
	def __init__(self, ident, callable_):
		self._ident = ident
		self._callable = callable_

	@property
	def ident(self):
		return self._ident

	def __call__(self, *args, **kwargs):
		return self._callable(*args, **kwargs)

	def unpack(self):
		return self._callable

	@staticmethod
	def chain(callables):
		ident = tuple([x.ident for x in callables])
		unpacked = [x.unpack() for x in callables]
		return CachableCallable(ident, chain(unpacked))


class RewrittenDict:
	def __init__(self, base, chg):
		self._base = base
		self._chg = chg

	def get(self, k, default=None):
		v = self._chg.get(k)
		if v is not None:
			return v
		else:
			return self._base.get(k, default)

	def __getitem__(self, k):
		v = self._chg.get(k)
		if v is not None:
			return v
		else:
			return self._base[k]


def make_rewrite(rules):
	if rules is None:
		return lambda t: t

	def f(t):
		t_new = dict()
		for k, v in rules.items():
			x = v.get(t[k])
			if x is not None:
				t_new[k] = x
		return RewrittenDict(t, t_new) if t_new else t

	return f


def make_ignore(rules):
	if rules is None:
		return lambda t: False

	def f(t):
		for k, v in rules.items():
			if t[k] in v:
				return True
		return False

	return f


class TextNormalizer:
	def __init__(self):
		self._f = []

	def add(self, name, f):
		self._f.append(CachableCallable(name, f))

	def to_callable(self):
		return CachableCallable.chain(self._f)

	def lower(self):
		self.add('lower', lambda s: s.lower())

	def strip(self):
		self.add('strip', lambda s: s.strip())

	def sub(self, pattern=r"\W", replacement=""):
		c_pattern = re.compile(pattern)
		self.add(
			('sub', pattern, replacement),
			lambda s: c_pattern.sub(replacement, s))

	def filter(self, k):
		self.add(('filter', k), lambda s: s if getattr(s, k)() else None)


class TokenNormalizer:
	def token_to_token(self, token):
		raise NotImplementedError()

	def token_to_text(self, text, token):
		raise NotImplementedError()


class SimpleTokenNormalizer(TokenNormalizer):
	def __init__(self, rewrite=None, ignore=None):
		self._rewrite = make_rewrite(rewrite)
		self._ignore = make_ignore(ignore)

	def token_to_token(self, token):
		token = self._rewrite(token)
		if self._ignore(token):
			return None
		return token

	def token_to_text(self, text, token):
		return text[token["start"]:token["end"]]


def normalizer_dict(normalizers):
	data = {}
	for x in normalizers:
		if isinstance(x, TextNormalizer):
			k = 'text'
		elif isinstance(x, TokenNormalizer):
			k = 'token'
		else:
			raise ValueError(f"illegal normalizer {x}")
		if k in data:
			raise ValueError(f"duplicate {k}")
		data[k] = x
	if len(data) < 2:
		raise ValueError("missing normalizers")
	return data


def default_normalizers():
	# Vectorian's default token mappings. You might want
	# to adjust this by adding lowercase mapping and/or
	# other pos tag mappings.

	txt = TextNormalizer()
	txt.sub(r"\W", "")
	txt.filter("isalpha")

	tok = SimpleTokenNormalizer(
		rewrite={
			# rewrite PROPN as NOUN to fix accuracy

			'pos': {
				'PROPN': 'NOUN'
			},
			'tag': {
				'NNP': 'NN',
				'NNPS': 'NNS',
			}
		},
		ignore={
			'pos': ['PUNCT']
		}
	)

	return [txt, tok]
