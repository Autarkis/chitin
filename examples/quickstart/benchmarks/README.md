# Sample quality acceptance

`sample-quality.json` turns the three bundled GLBs into stable browser/WASM
acceptance cases. `npm run benchmark:samples` compiles each sample in pinned
Chromium, measures the emitted hull artifact, and prints one
`QUALITY_BENCHMARK` JSON record per sample.

The benchmark deliberately separates two concepts:

- `regression` is the CI boundary. It prevents current geometry, hull-count,
  and triangle-count behavior from silently getting worse.
- `target` is the product-quality destination. Targets may remain unmet while
  the allocator is improved; the benchmark prints every remaining target gap.

The initial thresholds were recorded from the current deterministic WASM
runtime. They are not transferable performance numbers: compile time is
reported for diagnosis but is not gated across different machines.

The dish runs at the coarsest UI setting and additionally verifies that its
actual hull count reaches an effective budget below the 128-hull fine-detail
ceiling. This catches the regression where hollow-shell protection flattened
the entire detail slider to one effective configuration.

Surface coverage uses area-weighted samples and a tolerance equal to 2% of
each connected component's diagonal. This makes the metric scale-aware without
letting a large scene hide an omitted small part. Acceptance gates use detailed
component coverage separately from intentionally simplified decoration, so a
cheap olive or fin does not force the same fit policy as a dish or body.

Raw false fill estimates collider-occupied space outside the source solid. Thin
shell colliders inherently have substantial raw false fill, so cavity acceptance
uses `deep_false_fill_fraction`: occupied free space more than 2% of that
component's diagonal from its surface. This distinguishes normal collider
thickness from a hull that bridges a plate or slices through a bowl. All metrics
are sampled estimates, not exact proofs, and do not change the compilation
verdict from `not_evaluated`.
