import { describe, expect, it } from "vitest";

import {
  COMPILATION_REPORT_VERSION,
  createCompilationReport,
  validateCompilationReport,
} from "../src/report.js";
import type { ConvexHull } from "../src/types.js";

const HULLS: ConvexHull[] = [
  {
    vertices: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]),
  },
];

const RUNTIME = {
  kind: "browser_wasm",
  implementation: "@autarkis/chitin-lite",
  version: "0.2.0",
  compiler_version: "0.2.0+coacd1.0.11+emscripten5.0.7",
  dependencies: { coacd: "1.0.11", emscripten: "5.0.7" },
};

describe("CompilationReport", () => {
  it("builds the versioned browser shape without implying acceptance", () => {
    const report = createCompilationReport({
      profile: "walkable",
      input: {
        kind: "glb",
        source_vertices: 4,
        processed_vertices: 4,
        mesh_vertices: 4,
      },
      hulls: HULLS,
      phys_bytes: 128,
      deterministic: true,
      artifact_sha256: "a".repeat(64),
      runtime: RUNTIME,
      metrics: {
        source_surface_coverage: { value: 0.98, unit: "ratio", status: "measured" },
      },
    });

    expect(report.report_version).toBe(COMPILATION_REPORT_VERSION);
    expect(report.profile).toBe("walkable");
    expect(report.verdict.status).toBe("not_evaluated");
    expect(report.output).toMatchObject({
      hull_count: 1,
      vertex_count: 4,
      triangle_count: 4,
      byte_length: 128,
    });
    expect(report.metrics.coacd_deterministic).toEqual({
      value: true,
      unit: "boolean",
      status: "measured",
    });
    expect(report.metrics.source_surface_coverage).toEqual({
      value: 0.98,
      unit: "ratio",
      status: "measured",
    });
    expect(report.reproducibility.scope).toBe("same_runtime_toolchain");
    expect(validateCompilationReport(report)).toEqual([]);
  });

  it("rejects unsupported report versions and missing fields", () => {
    const invalid = {
      report_version: 99,
      reproducibility: { scope: "cross_runtime" },
    };
    const problems = validateCompilationReport(invalid);
    expect(problems).toContain("missing field: verdict");
    expect(problems.some((problem) => problem.includes("report_version"))).toBe(true);
    expect(problems.some((problem) => problem.includes("same_runtime_toolchain"))).toBe(true);
  });
});
