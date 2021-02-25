python >= 3.7

https://github.com/src-d/wmd-relax
https://github.com/src-d/wmd-relax/blob/master/emd_relaxed.h

Close reading vs high level view: Vectorian is a good tool, as you can dive into
specific search results very easily and fast.

A plethora of methods available to the researcher, but it can be
cumbersome to try out these methods on an actual corpus of texts.

Static embeddings like fasttext [ref] and GloVe [ref] and contextual embeddings (e.g. BERT [ref]).
Alignment methods [ref], Word Mover's Distance (WMD) [ref] and sentence embeddings [ref]
Cosine metrics [ref] or modified metrics such as Improved Cosine (e.g. Zhu) [ref]

All these methods offer various advantages and disadvantages and can be combined
in a variety of ways. 

The Vectorian framework offers a way to perform some of these methods, such as
static embeddings with WMD, fast on even larger corpuses, while allowing configuration
to other and more state-of-the-art methods (that are often slower by several
magnitudes without extensive preprocessing though).

Simple to use interface that is simple enough for the occasional user as well as
well as versatile enough for an experienced user who wants to quickly try out some
method on some corpus.

We understand the Vectorian as a generalizing framework that allows users to 
get a quick glimpse of different models and methods.

