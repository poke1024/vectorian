# manual installation

conda create -f environment.yml
conda activate vectorian
python setup.py install

python -m spacy download en_core_web_sm

this will give you a fully functional version of the vectorian API.
depending on what you want to use you also want to install:

## jupyterlab

jupyterlab>=3.0.7
ipywidgets>=7.6.3
matplotlib>=3.3.4

## flow visualizations

holoviews>=1.14.2
bokeh>=2.3.0

# design goals

started out as a high-performance system for interactive searches

allow interactive brute-force (index-free) searches using short
queries over sentences in a larger number of documents. the design
is geared towards high performance when search a large number of
small units (sentences) as opposed to comparing whole documents.

it's geared towards:
* fast loading of large corpus
* interactive searches that can vary a larger amount of parameters
without the need of anewed preprocessing the whole data

this is achieved by
(1) a highly optimized C++ core
(2) an efficient storage of document data that allows to load large
number of document data into memory without the need for disk access
(using intelling caching)

# changes over last version

## features (new)

* full support for fasttext (ngram-based construction)
* support for compressed fasttext embeddings
* support for any gensim-based key-vector embeddings
* support for contextual embeddings (spaCy transformers)
* support for custom PCA-compressed contextual embeddings
* custom vector space metrics in python
* added word mover's distance in various variants
* added word rotator's distance
* added needleman-wunsch and waterman-smith
* completely redesigned token normalization pipeline
* completely redesigned document storage and caching architecture
* support for text and matrix visualizations

# technical (new)

* switched from Eigen to xtensor
* prospective support for GPU-based vector operations via cupy
* removed pyarrow and the use of apache parquet in favor of h5py
* support for prebuilt linux wheels

# more



OPENBLAS="$(brew --prefix openblas)" jupyter lab

conda install -c conda-forge xtl
conda install -c conda-forge xtensor
conda install -c conda-forge xsimd
conda install -c conda-forge xtensor-blas
conda install -c conda-forge xtensor-python


# fasttext

let's get this right:
class Embedding:
    @property
    def n_dims(self):
        return 300
    def get_embeddings(tokens, array):
        # fill

# Important Related Packages
https://github.com/christianj6/binarized-word-movers-distance
https://github.com/flairNLP/flair


# Textdaten

http://martinweisser.org/corpora_site/diy_corpora.html

https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/3039
https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/3039/3039.xml?sequence=9&isAllowed=y

https://github.com/foonathan/memory

https://github.com/greg7mdp/sparsepp
https://www.etlcpp.com/documentation.html#

# offene Fragen

* unsere WMD/WRT Implementation mit anderen überprüfen
z.B. gensim, vgl. https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
https://pypi.org/project/wmd/
https://github.com/mkusner/wmd
https://www.ibm.com/blogs/research/2018/11/word-movers-embedding/
* die standard test sets aus dem WRT paper rechnen / Infrastruktur dafür

vectorian as standarized framework for evaluation (preprocessing etc.)