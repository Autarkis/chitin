# Adversarial Predicate Testing

Chitin's boundary-directed search is a development adversary for plane
classification. It is deliberately separate from frozen holdout evaluation:
developers may run it repeatedly, inspect every result, and promote minimized
failures into regression corpora. Its output must never be presented as an
unseen holdout result.

## What it tests

`chitin.f32_adversarial` generates finite vertex clouds in a range of binary
scales. It selects one vertex, places a plane one to four representable f64
or f32 steps away, on the vertex, or within fractions of a normalized grid
cell, and compares three answers:

1. The exact rational sign of the supplied IEEE-754 inputs.
2. The ordinary NumPy f64 dot-product sign.
3. The selected Chitin f32 policy.

The exact-input oracle does not claim that a source model represents ideal
real-number geometry. It establishes the exact sign of the binary numbers that
actually reached the predicate, avoiding the assumption that f64 is infallible
near zero.

Each case also exercises metamorphic properties that should preserve the
classification whenever the transformed exact inputs preserve it:

- vertex permutation;
- translation of vertices and plane point together;
- uniform scaling by an exact power of two.

Interesting cases are scored by exact disagreement count, metamorphic failures,
distance to the decision boundary, and ambiguity-path coverage. A
MAP-Elites-style archive retains only the strongest case in each boundary,
normal, scale, predicate-path, and failure-family bucket instead of accumulating
thousands of equivalent cases. Exact disagreements are greedily reduced to the
smallest vertex cloud that still reproduces the failure.

## Run it

```console
python scripts/search_f32_adversaries.py \
  --seed 0 \
  --cases 10000 \
  --output-dir artifacts/f32-adversaries/seed-0
```

The output directory must not already exist. Each minimized NPZ file has a
SHA-256 entry in `manifest.json`, along with the generator version, policy
version, seed, failure types, and reduction counts.

## Interpretation

This first layer targets classification predicates. It does not yet generate
watertight meshes or score clip/cap topology. The next layer should combine a
manifold mesh grammar with topology-aware coverage buckets and reuse the same
exact-oracle, metamorphic, reduction, and immutable-output conventions.
