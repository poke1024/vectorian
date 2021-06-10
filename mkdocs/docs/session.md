# Session

## What is a Session?

A `Session` brings together a fixed set of `Document`s and a fixed set of
`Embedding`s. It is the basis for creating an `Index`.

Creating a `Session` performs two important preprocessing steps of the data:

* all text and embedding data is mapped to a fixed vocabulary
* tokens are normalized according to given rules

Establishing a fixed vocabulary (from `Document`s and `Embedding`s) allows
the Vectorian to build data structures that are highly optimized for ensuing
operations.

More details on token normalization are given in the following section.

## Token and Tag Normalization

A small but important parameter during `Session` creation is the `normalizers`
option (which is usually set to "default"). 

* normalizing tokens on a string level, e.g. lowercasing all tokens
* ignoring certain tokens from all subsequent operations
* unifying or mapping token POS tags

It is in these settings that users declare whether two tokens like "the"
and "The" should be regarded identical or not. If they are to be unified,
the selection of embedding vectors depends on the configured embedding's
`sampling` setting (see [Embeddings](embeddings.md)).

## The Default Normalization

The Vectorian's default normalization is applies two sets of operations.

On the text level:

* all non-word characters are removed from tokens (e.g. "has-" becomes "has")
* if tokens do not contain at least one letter, they are ignored

On the POS tag level:

* tokens with POS tag "PROPN" are mapped to POS tag "NOUN"
* tokens with POS tag "PUNCT" (i.e. punctuation) are ignored

The motivation for rewriting "PROPN" tags is that these often pose a problem
for tag-weighted alignments due to their rather high inaccuracy.

## LabSession

`LabSession` is a specialization of `Session` that is specifically geared
towards use in `Jupyter`. It offers the following advantages over `Session`:

* displays a progress bar widget during performing queries
* `Result`s know how to render themselves in Jupyter as HTML
