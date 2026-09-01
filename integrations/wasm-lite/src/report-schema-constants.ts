// Generated from docs/compilation-report.schema.json; do not edit.

export const TOP_LEVEL_REQUIRED = [
  "report_version",
  "status",
  "profile",
  "verdict",
  "input",
  "output",
  "timings_ms",
  "warnings",
  "metrics",
  "processing",
  "runtime",
  "reproducibility",
  "config",
  "artifacts",
  "build_identity",
  "topology",
] as const;
export const VERDICT_REQUIRED = ["profile", "status", "reasons", "checks"] as const;
export const CHECK_REQUIRED = ["code", "status", "message"] as const;
export const INPUT_REQUIRED = ["kind", "source_vertices", "processed_vertices", "mesh_vertices"] as const;
export const OUTPUT_REQUIRED = [
  "collider_kind",
  "hull_count",
  "vertex_count",
  "triangle_count",
  "lod_tier_count",
  "byte_length",
] as const;
export const WARNING_REQUIRED = ["code", "severity", "message", "context"] as const;
export const METRIC_REQUIRED = ["value", "unit", "status"] as const;
export const PROCESSING_REQUIRED = ["pipeline", "fallbacks", "refinements"] as const;
export const FALLBACKS_REQUIRED = ["planar_substitute_hulls"] as const;
export const REFINEMENTS_REQUIRED = ["snug_fit"] as const;
export const SNUG_FIT_REQUIRED = ["status", "refined_hulls", "rejected_hulls", "skipped_hulls"] as const;
export const RUNTIME_REQUIRED = ["kind", "implementation", "version", "compiler_version", "dependencies"] as const;
export const REPRODUCIBILITY_REQUIRED = ["scope", "deterministic", "artifact_sha256"] as const;
export const CONFIG_REQUIRED = ["requested", "effective"] as const;
export const BUILD_IDENTITY_REQUIRED = [
  "effective_input_digest",
  "algorithm_version",
  "numerical_policy_version",
  "config_digest",
] as const;
export const TOPOLOGY_REQUIRED = [
  "component_count",
  "boundary_edge_count",
  "non_manifold_edge_count",
  "degenerate_face_count",
  "consistently_oriented",
  "closed",
  "two_manifold",
] as const;

export const REPORT_STATUS_VALUES = ["complete", "rejected"] as const;
export const VERDICT_STATUS_VALUES = ["pass", "fail", "not_evaluated"] as const;
export const CHECK_STATUS_VALUES = ["pass", "fail", "not_evaluated"] as const;
export const WARNING_SEVERITY_VALUES = ["info", "warning", "error"] as const;
export const METRIC_STATUS_VALUES = ["measured", "not_measured", "not_applicable"] as const;
export const SNUG_FIT_STATUS_VALUES = ["applied", "skipped", "not_requested", "unknown"] as const;
