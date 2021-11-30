# Index

## What is it for?

In order to perform actual searches on `Span`s, i.e. `Document` portions,
you need to create an `Index`. The most important method in an `Index` is
the `find` method, which returns results, as in the following example:

```
my_index.find("railway museum", n=1)
```

Why does it need an `Index`? For some kinds of searches, this allows the
Vectorian to perform various optimizations under the hood - quite similar
to an index in a database system. Certain kinds of `Index` objects that are
expensive to create can also be saved and loaded, but this is beyond the
scope of this introduction.

## Constructing an `Index`

An `Index` is created from two components, a `Partition` and a `SpanSim`:

* The given `Partition` indicates the granularity of search and which items
should get indexed for searching (see the section on [Documents](../documents)
for more details). In short, `Partition` models how to create `Span`s from
`Document`s.
* The `SpanSim` models the approach taken to compute the similarity
of two `Span`s (e.g. a specific sort of alignment). See the section on
[Span Similarity](../sim_span) for more details.

Here is an example (`my_span_sim` is an instance of `SpanSim`):

```
my_index = session.partition("document").index(my_span_sim, nlp)
```
