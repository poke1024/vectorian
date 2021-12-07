import re
import collections
import sklearn
import gensim
import gensim.models
import gensim.downloader
import numpy as np
import requests
import zipfile
import urllib.parse
import os

from pathlib import Path
from vectorian.tqdm import tqdm


def gensim_version():
	return int(gensim.__version__.split(".")[0])


_custom_cache_path = os.environ.get("VECTORIAN_CACHE_HOME")


def set_cache_path(path):
	global _custom_cache_path
	_custom_cache_path = Path(path)


def make_cache_path():
	if _custom_cache_path is None:
		cache_path = Path.home() / ".vectorian" / "embeddings"
	else:
		cache_path = Path(_custom_cache_path) / "embeddings"
	cache_path.mkdir(exist_ok=True, parents=True)
	return cache_path


def extraction_tqdm(tokens, name):
	return tqdm(tokens, desc=f"Extracting {name}", disable=len(tokens) < 5000)


def download(url, path, force_download=False):
	path = Path(path)
	download_path = path / urllib.parse.urlparse(url).path.split("/")[-1]
	is_zip = download_path.suffix == ".zip"

	if is_zip:
		result_path = path / download_path.stem
	else:
		result_path = download_path

	if result_path.exists() and not force_download:
		return result_path

	with tqdm(desc="Downloading " + url, unit='iB', unit_scale=True) as pbar:
		response = requests.get(url, stream=True)

		total_length = int(response.headers.get('content-length', 0))
		pbar.reset(total=total_length)

		try:
			with open(download_path, "wb") as f:
				for data in response.iter_content(chunk_size=4096):
					pbar.update(len(data))
					f.write(data)
		except:
			download_path.unlink(missing_ok=True)
			raise

	if download_path != result_path:
		extracted = []
		with zipfile.ZipFile(download_path, 'r') as zf:
			for zi in zf.infolist():
				if zi.filename[-1] == '/':
					continue
				zi.filename = os.path.basename(zi.filename)
				p = zf.extract(zi, result_path.parent)
				extracted.append(Path(p))

		if len(extracted) == 1:
			extracted[0].rename(result_path)

		download_path.unlink()

	return result_path if result_path.exists() else None


def normalize_word2vec(name, tokens, embeddings, normalizer, sampling='nearest'):
	if sampling not in ('nearest', 'average'):
		raise ValueError(f'Expected "nearest" or "average", got "{sampling}"')

	embeddings = embeddings.astype(np.float32)

	f_mask = np.zeros((embeddings.shape[0],), dtype=np.bool)
	f_tokens = []
	token_to_ids = dict()

	for i, t in enumerate(tqdm(tokens, desc=f"Normalizing tokens in {name}")):
		nt = normalizer(t)
		if nt is None:
			continue
		if sampling != 'average' and nt != t:
			continue
		indices = token_to_ids.get(nt)
		if indices is None:
			token_to_ids[nt] = [i]
			f_tokens.append(nt)
			f_mask[i] = True
		else:
			indices.append(i)

	if sampling == 'average':
		for indices in tqdm(token_to_ids.values(), desc=f"Merging tokens in {name}", total=len(token_to_ids)):
			if len(indices) > 1:
				i = indices[0]
				embeddings[i] = np.mean(embeddings[indices], axis=0)

	f_embeddings = embeddings[f_mask]
	embeddings = None

	assert f_embeddings.shape[0] == len(f_tokens)

	return f_tokens, f_embeddings


def load_glove_txt(csv_path):
	tokens = []
	with open(csv_path, "r") as f:
		text = f.read()

	lines = text.split("\n")
	n_rows = len(lines)
	n_cols = len(lines[0].strip().split()) - 1

	embeddings = np.empty(
		shape=(n_rows, n_cols), dtype=np.float32)

	for line in tqdm(lines, desc="Importing " + str(csv_path)):
		values = line.strip().split()
		if values:
			t = values[0]
			if t:
				embeddings[len(tokens), :] = values[1:]
				tokens.append(t)

	embeddings = embeddings[:len(tokens), :]

	return tokens, embeddings


def extract_numberbatch(path, languages):
	# e.g. extract_numberbatch("/path/to/numberbatch-19.08.txt", ["en", "de"])
	# then use KeyedVectors.load()

	path = Path(path)
	languages = set(languages)

	pattern = re.compile(r"^/c/([a-z]+)/")

	with open(path, "r") as f:
		num_lines, num_dimensions = [int(x) for x in f.readline().split()]
		vectors = collections.defaultdict(lambda: {
			"keys": [],
			"vectors": []
		})

		for _ in tqdm(range(num_lines)):
			line = f.readline()
			m = pattern.match(line)
			if m:
				lang = m.group(1)
				if lang in languages:
					line = line[len(m.group(0)):]
					cols = line.split()
					key = cols[0]
					if key.isalpha():
						record = vectors[lang]
						record["keys"].append(key)
						record["vectors"].append(
							np.array([float(x) for x in cols[1:]]))

	for lang, record in vectors.items():
		wv = gensim.models.KeyedVectors(num_dimensions)
		wv.add_vectors(record["keys"], record["vectors"])
		wv.save(str(path.parent / f"{path.stem}-{lang}.kv"))


def compress_keyed_vectors(model, n_dims):
	pca = sklearn.decomposition.PCA(n_components=n_dims)
	vectors = pca.fit_transform(model.vectors)
	wv = gensim.models.KeyedVectors(
		vector_size=n_dims, count=vectors.shape[0])
	if gensim_version() < 4:
		for i, k in enumerate(model.vocab):
			wv.add_vector(k, vectors[i])
	else:
		for k, i in model.key_to_index.items():
			wv.add_vector(k, vectors[i])
	return wv
