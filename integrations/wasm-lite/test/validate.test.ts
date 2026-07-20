import { describe, it, expect } from "vitest";

import { validateMeshInput, validateConfig, ChitinError } from "../src/index.js";

const V = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]); // 3 valid verts
const F = new Int32Array([0, 1, 2]);

function codeOf(fn: () => void): string {
  try {
    fn();
  } catch (e) {
    return e instanceof ChitinError ? e.code : `not-ChitinError:${e}`;
  }
  return "no-throw";
}

describe("validateMeshInput", () => {
  it("accepts a valid mesh", () => {
    expect(() => validateMeshInput(V, F)).not.toThrow();
  });

  it("rejects empty vertices/faces", () => {
    expect(codeOf(() => validateMeshInput(new Float64Array(), F))).toBe("INVALID_MESH");
    expect(codeOf(() => validateMeshInput(V, new Int32Array()))).toBe("INVALID_MESH");
  });

  it("rejects a vertex length that is not a multiple of 3", () => {
    expect(() => validateMeshInput(new Float64Array([0, 0]), F)).toThrow(/multiple of 3/);
  });

  it("rejects a NaN vertex coordinate", () => {
    const bad = new Float64Array([0, 0, 0, NaN, 0, 0, 0, 1, 0]);
    expect(() => validateMeshInput(bad, F)).toThrow(/not finite/);
    expect(codeOf(() => validateMeshInput(bad, F))).toBe("INVALID_MESH");
  });

  it("rejects an Infinity vertex coordinate", () => {
    const bad = new Float64Array([0, 0, 0, Infinity, 0, 0, 0, 1, 0]);
    expect(() => validateMeshInput(bad, F)).toThrow(/not finite/);
  });

  it("rejects an out-of-range face index (99 for a 3-vertex mesh)", () => {
    expect(() => validateMeshInput(V, new Int32Array([0, 1, 99]))).toThrow(/out of range/);
  });

  it("rejects a negative face index", () => {
    expect(() => validateMeshInput(V, new Int32Array([0, -1, 2]))).toThrow(/out of range/);
  });
});

describe("validateConfig", () => {
  it("accepts an empty config and valid values", () => {
    expect(() => validateConfig({})).not.toThrow();
    expect(() =>
      validateConfig({ threshold: 0.05, maxConvexHull: -1, prepResolution: 50, mctsNodes: 20 }),
    ).not.toThrow();
  });

  it("rejects out-of-range threshold", () => {
    expect(codeOf(() => validateConfig({ threshold: 0 }))).toBe("INVALID_CONFIG");
    expect(() => validateConfig({ threshold: 1.5 })).toThrow(/threshold/);
    expect(() => validateConfig({ threshold: NaN })).toThrow(/threshold/);
  });

  it("rejects invalid maxConvexHull", () => {
    expect(() => validateConfig({ maxConvexHull: 0 })).toThrow(/maxConvexHull/);
    expect(() => validateConfig({ maxConvexHull: -2 })).toThrow(/maxConvexHull/);
  });

  it("enforces CoACD's prepResolution bounds [5, 1000]", () => {
    expect(() => validateConfig({ prepResolution: 4 })).toThrow(/prepResolution/);
    expect(() => validateConfig({ prepResolution: 1001 })).toThrow(/prepResolution/);
    expect(() => validateConfig({ prepResolution: 5 })).not.toThrow();
    expect(() => validateConfig({ prepResolution: 1000 })).not.toThrow();
  });

  it("rejects non-positive-integer options", () => {
    expect(() => validateConfig({ sampleResolution: -1 })).toThrow(/sampleResolution/);
    expect(() => validateConfig({ mctsNodes: 2.5 })).toThrow(/mctsNodes/);
  });
});
