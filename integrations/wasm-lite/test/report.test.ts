import { readFileSync } from "node:fs";

import Ajv2020 from "ajv/dist/2020.js";
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

function reportFixture() {
  return createCompilationReport({
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
}

describe("CompilationReport", () => {
  it("builds the versioned browser shape without implying acceptance", () => {
    const report = reportFixture();

    expect(report.report_version).toBe(COMPILATION_REPORT_VERSION);
    expect(report.profile).toBe("walkable");
    expect(report.verdict.status).toBe("not_evaluated");
    expect(report.output).toMatchObject({
      hull_count: 1,
      vertex_count: 4,
      triangle_count: 4,
      byte_length: 128,
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

  it("matches the canonical cross-runtime JSON Schema", () => {
    const schema = JSON.parse(
      readFileSync(new URL("../../../docs/compilation-report.schema.json", import.meta.url), "utf8"),
    );
    const validate = new Ajv2020({ strict: false }).compile(schema);
    const report = reportFixture();
    expect(validate(report), JSON.stringify(validate.errors, null, 2)).toBe(true);
  });
});

// Keep the dependency-free validator aligned with schema-required fields.
describe("validateCompilationReport covers the schema's required fields", () => {
  const schema = JSON.parse(
    readFileSync(new URL("../../../docs/compilation-report.schema.json", import.meta.url), "utf8"),
  );

  function collectRequiredPaths(node: any, prefix: string[] = []): string[][] {
    const paths: string[][] = [];
    if (Array.isArray(node?.required)) {
      for (const key of node.required) paths.push([...prefix, key]);
    }
    if (node?.properties) {
      for (const [key, sub] of Object.entries<any>(node.properties)) {
        paths.push(...collectRequiredPaths(sub, [...prefix, key]));
      }
    }
    return paths;
  }

  function deleteAtPath(target: any, path: string[]): void {
    const parent = path.slice(0, -1).reduce((acc, key) => acc?.[key], target);
    if (parent && typeof parent === "object") delete parent[path[path.length - 1]];
  }

  it("catches a missing field for every required property in the schema's object tree", () => {
    const paths = collectRequiredPaths(schema);
    // Include nested objects, not only the 14 top-level keys.
    expect(paths.length).toBe(52);

    for (const path of paths) {
      const report = reportFixture() as any;
      deleteAtPath(report, path);
      const problems = validateCompilationReport(report);
      expect(problems.length, `expected a problem when ${path.join(".")} is missing`).toBeGreaterThan(0);
    }
  });

  it("catches a missing field inside warnings[], metrics{}, and verdict.checks[] items", () => {
    // Array items and map values are outside the `properties` walk.
    for (const key of schema.$defs.warning.required as string[]) {
      const report = reportFixture() as any;
      report.warnings = [{ code: "W", severity: "info", message: "m", context: {} }];
      delete report.warnings[0][key];
      expect(validateCompilationReport(report).length, `warnings[].${key}`).toBeGreaterThan(0);
    }
    for (const key of schema.$defs.metric.required as string[]) {
      const report = reportFixture() as any;
      const [name] = Object.keys(report.metrics);
      delete report.metrics[name][key];
      expect(validateCompilationReport(report).length, `metrics{}.${key}`).toBeGreaterThan(0);
    }
    for (const key of schema.properties.verdict.properties.checks.items.required as string[]) {
      const report = reportFixture() as any;
      report.verdict.checks = [{ code: "c", status: "pass", message: "m" }];
      delete report.verdict.checks[0][key];
      expect(validateCompilationReport(report).length, `verdict.checks[].${key}`).toBeGreaterThan(0);
    }
  });
});
