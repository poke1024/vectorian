# The Vectorian

The Vectorian is a text search engine based on word embeddings and
alignments. Its main intention is academic research and teaching purposes.

API documentation can be found under https://poke1024.github.io/vectorian-2021/index.html

## Design Goals

The Vectorian started out as a high-performance system for interactive searches.

It allow interactive brute-force (index-free) searches using short
queries over sentences in a larger number of documents. The design
is geared towards high performance when search a large number of
small units (sentences) as opposed to comparing whole documents:

* fast loading of large corpus
* interactive searches that can vary a larger amount of parameters
without the need of anewed preprocessing the whole data

This is achieved by:

* a highly optimized C++17 core
* an efficient storage of document data that allows to load large
number of document data into memory without the need for disk access
(using intellingent caching)

## Manual Installation

```
conda create -f environment.yml
conda activate vectorian
python setup.py install

python -m spacy download en_core_web_sm
```

this will give you a fully functional version of the vectorian API.
depending on what you want to use you might also want to install:

### Jupyterlab

```
jupyterlab>=3.0.7
ipywidgets>=7.6.3
matplotlib>=3.3.4
```

### Flow Visualizations

```
holoviews>=1.14.2
bokeh>=2.3.0
```

## Changes over previous web-based version (pre 2021)

### Features

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

### Technical

* switched from Eigen to xtensor
* prospective support for GPU-based vector operations via cupy
* removed pyarrow and the use of apache parquet in favor of h5py
* support for prebuilt Linux wheels
