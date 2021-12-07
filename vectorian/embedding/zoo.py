from functools import partial
from .utils import download, make_cache_path


def _zenodo_url(record, name):
	return f'https://zenodo.org/record/{record}/files/{name}'


class Zoo:
	_initialized = False

	_numberbatch_lang_codes = [
		'af', 'ang', 'ar', 'ast', 'az', 'be', 'bg', 'ca', 'cs', 'cy', 'da',
		'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fo',
		'fr', 'fro', 'ga', 'gd', 'gl', 'grc', 'gv', 'he', 'hi', 'hsb', 'hu',
		'hy', 'io', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'ku', 'la', 'lt', 'lv',
		'mg', 'mk', 'ms', 'mul', 'nl', 'no', 'non', 'nrf', 'nv', 'oc', 'pl',
		'pt', 'ro', 'ru', 'rup', 'sa', 'se', 'sh', 'sk', 'sl', 'sq', 'sv', 'sw',
		'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'vo', 'xcl', 'zh'
	]

	_embeddings = {
	}

	@staticmethod
	def _init():
		if Zoo._initialized:
			return

		from fasttext.util.util import valid_lang_ids as ft_valid_lang_ids
		for lang in ft_valid_lang_ids:
			Zoo._embeddings[f'fasttext-{lang}'] = {
				'constructor': PretrainedFastText,
				'lang': lang
			}

			Zoo._embeddings[f'fasttext-{lang}-mini'] = {
				'constructor': CompressedFastTextVectors,
				'url': _zenodo_url(4905385, f'fasttext-{lang}-mini')
			}

		for lang in Zoo._numberbatch_lang_codes:
			Zoo._embeddings[f'numberbatch-19.08-{lang}'] = {
				'constructor': partial(Word2VecVectors, binary=True),
				'url': _zenodo_url(4911598, f'numberbatch-19.08-{lang}.zip'),
				'name': f'numberbatch-19.08-{lang}'
			}

		for d in [50, 100, 200, 300]:
			Zoo._embeddings[f'glove-6B-{d}'] = {
				'constructor': partial(Word2VecVectors, binary=True),
				'url': _zenodo_url(4925376, f'glove.6B.{d}d.zip'),
				'name': f'glove-6B-{d}'
			}

		for name, sizes in {
			'42B': [300],
			'840B': [300],
			'twitter.27B': [25, 50, 100, 200]}.items():

			for size in sizes:
				Zoo._embeddings[f'glove-{name}-{size}'] = {
					'constructor': PretrainedGloVe,
					'name': name,
					'ndims': size
				}

		Zoo._initialized = True

	@staticmethod
	def _download(url, force_download=False):
		cache_path = make_cache_path()

		download_path = cache_path / "models"
		download_path.mkdir(exist_ok=True, parents=True)

		return download(url, download_path, force_download=force_download)

	@staticmethod
	def list():
		Zoo._init()
		return tuple(sorted(Zoo._embeddings.keys()))

	@staticmethod
	def load(name, force_download=False):
		Zoo._init()
		spec = Zoo._embeddings.get(name)
		if spec is None:
			raise ValueError(f"unknown embedding name {name}")
		kwargs = dict((k, v) for k, v in spec.items() if k not in ("constructor", "url"))
		if "url" in spec:
			kwargs["path"] = Zoo._download(spec["url"], force_download)
		return spec["constructor"](**kwargs)
