# Compilation report contract

`CompilationReport` is the versioned, JSON-safe result contract shared by the
Python compiler and `@autarkis/chitin-lite`. Its schema is
[`compilation-report.schema.json`](compilation-report.schema.json).

The Python service returns it under `compilation_report`, alongside the flat
`report.json` fields. Bundle manifests carry the same object under
`quality.report`. The browser `compileGlb()` API returns the object
directly as `result.report`.

## Contract rules

- `report_version` changes only for a breaking shape or semantic change.
- Fields are snake_case so serialized Python and TypeScript objects are
  identical.
- Missing measurements stay present with `value: null` and a metric status of
  `not_measured` or `not_applicable`.
- A compiler that did not run the profile's artifact-level checks must return
  `verdict.status: "not_evaluated"`. It must not infer a pass from successful
  decomposition.
- Warning and check `code` values are stable machine-readable identifiers;
  `message` is user-facing text.
- `decomposition_failure_hulls` and `planar_substitute_hulls` are separate.
  The Python compiler populates `planar_substitute_hulls` as an integer,
  `0` when no substitution happened during the run, never `null` for a
  normal Python run. The browser compiler (`@autarkis/chitin-lite`) still
  reports `null`: it has no planar-substitution path.
- Refinements report `applied`, `skipped`, `not_requested`, or `unknown`.
  Requesting snug-fit without recorded execution statistics is `skipped`.
- Reproducibility is scoped to `same_runtime_toolchain`. Python-native and
  browser-WASM results are not required to have matching hulls or bytes.
- Browser interactive compilation reports
  `INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED` at `info` severity when its
  deterministic scene policy gives small connected parts one convex
  approximation. The thresholds, total hull budget, component counts, and
  worker and MCTS settings are recorded under `config.effective`; this is an
  intentional interactive approximation, not a decomposition-failure fallback.
- Browser interactive compilation reports `INTERACTIVE_THRESHOLD_CLAMPED` at
  `info` severity when a caller requests a detailed-part threshold below the
  configured interactive minimum. Requested and effective values remain
  explicit in the report; the clamp is never silent.
- Browser interactive compilation reports `INTERACTIVE_HOLLOW_SHELL_GUARD` at
  `info` severity when a coarse request is limited for low-occupancy shell
  components. The component count, occupancy cutoff, effective shell threshold,
  and reserved hull capacity are recorded under `config.effective`. This warning
  describes a planning heuristic and does not imply a geometric-fit or
  free-space-clearance pass.
- Browser interactive compilation reports `INTERACTIVE_IMPORTANCE_GUARD` when
  a coarse threshold is limited for a scene-dominant, spatially sparse connected
  component. The requested threshold, occupancy cutoff, guarded component count,
  and effective threshold range are recorded. This preserves useful body-level
  decomposition as a planning policy; it does not claim that silhouette or
  coverage passed a fit check.
- Browser interactive compilation reports `INTERACTIVE_HULL_VERTICES_ADAPTED`
  when per-hull vertex ceilings are assigned from scene-relative size and the
  component's isoperimetric quotient. Effective minimum, maximum, and mean caps
  are recorded under `config.effective`. The quotient is a planning proxy for
  geometric roundness, not an evaluated curvature or fit outcome.
- Browser artifact-fit sampling is opt-in through `compileGlb(..., { quality })`
  because it adds verification work. When enabled, the report records
  `source_surface_coverage`, `worst_component_surface_coverage`,
  detailed-component coverage, `collider_volume_precision`, raw
  `false_fill_fraction`, and clearance-aware `deep_false_fill_fraction` with
  deterministic Halton sample counts, method identity, strict volume tolerance,
  and per-component tolerances. Component metrics retain hull ownership and
  geometry/planning diagnostics. These are measured estimates;
  enabling them does not change `verdict.status` from `not_evaluated`.

## Compatibility

Readers should reject an unknown `report_version`, but tolerate new metric,
timing, warning-code, check-code, and artifact keys within version 1. Producers
must not remove required fields or change their meaning without incrementing
the version.

The canonical object deliberately carries both `status` and `verdict.status`:

- `status` describes whether compilation completed or was rejected.
- `verdict.status` describes whether the selected profile passed, failed, or
  was not evaluated.

A successful browser decomposition can therefore be `status: "complete"` and
`verdict.status: "not_evaluated"` until profile-specific probes run.
