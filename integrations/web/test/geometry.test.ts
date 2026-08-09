import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";

import {
  applyBindPose,
  calculateVertexBounds,
  combinePhysGeometry,
  materializePhysHulls,
  transformHullVertices,
} from "../src/geometry.js";
import { parsePhys, type PhysFile } from "../src/phys-parser.js";

const DIR = resolve(__dirname, "conformance");

function load(name: string): PhysFile {
  const buffer = readFileSync(resolve(DIR, name));
  return parsePhys(
    buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength,
    ),
  );
}

describe("engine-agnostic geometry", () => {
  it("materializes rigged hulls in asset-local bind pose", () => {
    const phys = load("rigged.phys");
    const stored = materializePhysHulls(phys, { pose: "stored" });
    const bound = materializePhysHulls(phys, { pose: "bind" });
    const child = phys.hulls.findIndex((hull) => hull.boneIndex === 1);

    expect(child).toBeGreaterThanOrEqual(0);
    for (let i = 0; i < stored.hulls[child].vertices.length; i += 3) {
      expect(bound.hulls[child].vertices[i]).toBeCloseTo(
        stored.hulls[child].vertices[i] + 1,
        6,
      );
      expect(bound.hulls[child].vertices[i + 1]).toBeCloseTo(
        stored.hulls[child].vertices[i + 1],
        6,
      );
      expect(bound.hulls[child].vertices[i + 2]).toBeCloseTo(
        stored.hulls[child].vertices[i + 2],
        6,
      );
    }
  });

  it("selects an LOD while materializing", () => {
    const phys = load("multi_lod.phys");
    expect(
      materializePhysHulls(phys, { lodConcavity: 0.9 }).hulls[0].vertices,
    ).toHaveLength(12);
  });

  it("applies scale, quaternion rotation, and translation in TRS order", () => {
    const vertices = new Float32Array([1, 0, 0]);
    const transformed = transformHullVertices(vertices, {
      scale: [2, 1, 1],
      rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
      translation: [10, 20, 30],
    });
    expect([...transformed]).toEqual([10, 22, 30]);
  });

  it("combines hull buffers and offsets indices", () => {
    const phys = load("rigged.phys");
    const geometry = materializePhysHulls(phys);
    const combined = combinePhysGeometry(geometry);
    const firstVertexCount = geometry.hulls[0].vertices.length / 3;
    const secondRange = combined.ranges[1];

    expect(combined.ranges).toHaveLength(geometry.hulls.length);
    expect(combined.indices[secondRange.indexStart]).toBe(
      geometry.hulls[1].indices[0] + firstVertexCount,
    );
    expect(combined.bounds).toEqual(geometry.bounds);
  });

  it("computes bounds and rejects malformed coordinates", () => {
    expect(
      calculateVertexBounds(new Float32Array([-1, 2, 4, 3, -2, 1])),
    ).toEqual({ min: [-1, -2, 1], max: [3, 2, 4] });
    expect(() => calculateVertexBounds(new Float32Array([1, 2]))).toThrow(
      /multiple of 3/,
    );
    expect(() => calculateVertexBounds(new Float32Array([1, NaN, 3]))).toThrow(
      /non-finite/,
    );
  });

  it("validates bind transforms at the generic boundary", () => {
    expect(() =>
      applyBindPose(new Float32Array([0, 0, 0]), new Float32Array(15)),
    ).toThrow(/16 finite/);
  });
});
