import fasttext.util
from gensim.models.fasttext import load_facebook_vectors
import compress_fasttext
from pathlib import Path
import download

# fasttext.util.download_model('de')
big_model = load_facebook_vectors(str(Path.home / "fasttext/cc.de.300.bin"))
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save(str(Path.home / "fasttext/fasttext-de-mini"))
