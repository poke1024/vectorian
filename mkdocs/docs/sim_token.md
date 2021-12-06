# Token Similarity

## EmbeddingTokenSim

`vectorian.sim.token.EmbeddingTokenSim` describes a strategy to compute a similarity
score from embeddings. It consists of two things:

* a specific embedding which serves as the source of vectors (e.g. fastText)
* a strategy to compute a scalar similarity value from two vectors

The first item is modelled by `vectorian.embedding.TokenEmbedding` in the Vectorian, the
latter by `vectorian.sim.vector.VectorSim`.

![Basic Operators implementing EmbeddingTokenSim](images/sim_token.png)

## VectorSim

`vectorian.sim.vector.VectorSim` is a strategy that describes how to
compute a scalar similarity value from two given embedding vectors. The resulting
value is expected to lie between 0 (which implies minimum similarity) and 1 (which
implies maximum similarity). Negative values will be clipped to 0 later in the
pipeline.

An obvious choice for `VectorSim` is `vectorian.sim.token.CosineSim`,
which computes the cosine of the angle between two embedding vectors.

Other `VectorSim` implementations based on other metrics are possible. It is
also possible to compute distances and later convert them to similarities (see
example further below).

The following diagram shows which `VectorSim`s are currently implemented
in the Vectorian.

![Basic Operators implementing EmbeddingTokenSim](images/sim_vector.png)

## Modifiers on VectorSim

Using `vectorian.sim.token.ModifiedVectorSim` and one or more
`vectorian.sim.kernel.UnaryOperator`s it is possible to perform additional
operations on a `VectorSim`.

Note that this kind of operations are always based eventually on *one* single
embedding, since all such computations boil down to one root `VectorSim`,
which usually operates on a single embedding.

For example the following code models a similarity based on the embedding stored
in `fastText`, where similarity between two vectors **u** and **v** is calculated
as `cos(phi) - 0.2`, if `phi` is the angle between **u** and **v**.

```
vectorian.sim.token.EmbeddingTokenSim(
    fastText,
    vectorian.sim.vector.ModifiedVectorSim(
        vectorian.sim.vector.CosineSim(),
        vectorian.sim.kernel.Bias(-0.2)
    ))
```

The currently available unary operators (like e.g. `Bias`) that can be used
with `ModifiedVectorSim` are shown in the following diagram:

![Kernels for modifying similarities](images/sim_kernel.png)

## Modifiers on EmbeddingTokenSim

The Vectorian also allows you to build completely new `EmbeddingTokenSim`
strategies that are capable of combining different `EmbeddingTokenSim` instances
that employ different embeddings.

One such example is `vectorian.sim.modifier.MixedTokenSimilarity`, which takes a
number of `EmbeddingTokenSim` instances and combines them in a weighted sum.

Here is an example that combines two different `EmbeddingTokenSim` instances to
build a new mixed `EmbeddingTokenSim`:

```
vectorian.sim.modifier.MixedTokenSimilarity(
	[
        vectorian.sim.token.EmbeddingTokenSim(
            fastText,
            vectorian.sim.vector.CosineSim(),
        ),
        vectorian.sim.token.EmbeddingTokenSim(
            glove,
            vectorian.sim.vector.ModifiedVectorSim(
                vectorian.sim.vector.CosineSim(),
                vectorian.sim.kernel.Bias(-0.2)
            )),
    ],
    weights=[0.3, 0.7])
```

Other combinators are possible. For example, `MaximumTokenSimilarity` takes the
highest similarity of a number of given `EmbeddingTokenSim`s for each token.
Accordingly, `MinimumTokenSimilarity` takes the lowest similarity.

The following diagram shows all such multi-embedding combinators which are
currently implemented:

![Classes implementing modified tokens similarity computations](images/sim_modifier.png)

## Distances and Similarities

Here is an example of using a Euclidean distance as a `VectorSim`:

```
vectorian.metrics.ModifiedVectorSim(
    vectorian.sim.token.PNormDistance(p=2),
    vectorian.sim.kernel.RadialBasis(gamma=2.5),
    vectorian.sim.kernel.DistanceToSimilarity()
)
```

Using `RadialBasis`, the distance `d` is first calibrated such that the range
[0, 1] contains a meaningful distance (with 1 being minimum similarity), then
`DistanceToSimilarity` is used to compute `1 - d`, i.e. to turn the distance
into a scalar similarity value.
