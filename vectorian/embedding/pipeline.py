class Stats:
	def __init__(self, name, dimension):
		self._name = name
		self._dimension = dimension

	@property
	def name(self):
		return self._name

	@property
	def dimension(self):
		return self._dimension


def stats_from_sentence_bert(nlp):
	try:
		import spacy_sentence_bert
	except ImportError:
		return None

	meta = nlp.meta
	dimension = meta.get('vectors', {}).get('width')

	sentence_bert = None
	for name, x in nlp.pipeline:
		if isinstance(x, spacy_sentence_bert.language.SentenceBert):
			if sentence_bert is not None:
				return None
			sentence_bert = x

	if not sentence_bert:
		return None

	model_name = sentence_bert.model_name

	lang = meta['lang']
	name = f"sentence-bert-{lang}-{model_name}"

	if dimension is None or dimension == 0:
		dimension = nlp("").vector.shape[0]

	return Stats(name, dimension)


def stats_from_meta(nlp):
	meta = nlp.meta
	vectors = meta.get('vectors')
	if vectors is None:
		return None

	dimension = vectors.get("width")
	name = vectors.get("name")
	if name is None:
		return None

	if dimension is None or dimension == 0:
		dimension = nlp("").vector.shape[0]

	return Stats(name, dimension)


decomposers = [
	stats_from_sentence_bert,
	stats_from_meta
]


def register_decomposer(f):
	decomposers.append(f)


def decompose_nlp(nlp):
	for f in decomposers:
		data = f(nlp)
		if data is not None:
			return data
	return None
