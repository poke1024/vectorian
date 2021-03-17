import fasttext.util
from gensim.models.fasttext import load_facebook_vectors
import compress_fasttext
from pathlib import Path
import download

if True:
	glove_name = "6B"

	txt_data_path = Path("/local/u/liebl/fasttext") / f"glove-{glove_name}"

	# url = "http://nlp.stanford.edu/data/glove.6B.zip"
	url = f"http://nlp.stanford.edu/data/glove.{glove_name}.zip"
	download.download(url, str(txt_data_path), kind="zip", progressbar=True)

	from gensim.scripts.glove2word2vec import glove2word2vec

	glove2word2vec(glove_input_file=str(txt_data_path / f"glove.{glove_name}.300d.txt"),
				   word2vec_output_file=str(txt_data_path / f"glove.{glove_name}.300d.word2vec.txt"))

	from gensim.models.keyedvectors import KeyedVectors

	glove_model = KeyedVectors.load_word2vec_format(str(txt_data_path / f"glove.{glove_name}.300d.word2vec.txt"),
													binary=False)

	small_model = compress_fasttext.prune_ft_freq(glove_model, pq=True)
	small_model.save("/local/u/liebl/fasttext/glove-6b-mini")

if False:
	# fasttext.util.download_model('de')
	big_model = load_facebook_vectors("/local/u/liebl/fasttext/cc.de.300.bin")
	small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
	small_model.save("/local/u/liebl/fasttext/fasttext-de-mini")
