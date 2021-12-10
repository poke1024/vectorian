from .utils import gensim_version

if gensim_version() >= 4:
	raise RuntimeError("Vectorian needs gensim < 4.0.0")

from .token.keyed import GensimVectors, Word2VecVectors, PretrainedGloVe, StackedEmbedding
from .token.fasttext import CompressedFastTextVectors, PretrainedFastText
from .token.contextual import ContextualEmbedding
from .span import SentenceEmbedding
