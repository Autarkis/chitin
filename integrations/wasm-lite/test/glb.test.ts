import { describe, expect, it } from "vitest";

import { parseGlb } from "../src/glb.js";
import { makeGlb, makeSparseGlb } from "./glb-fixture.js";

describe("parseGlb", () => {
  it("merges the active scene with transforms, instancing, indexed and unindexed primitives", () => {
    const mesh = parseGlb(makeGlb());
    expect(mesh).toMatchObject({
      mesh_count: 2,
      primitive_count: 4,
      node_count: 2,
    });
    expect(mesh.vertices).toHaveLength(12 * 3);
    expect(mesh.faces).toHaveLength(4 * 3);
    expect(Array.from(mesh.vertices.slice(0, 9))).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(mesh.vertices.slice(18, 27))).toEqual([10, 0, 0, 11, 0, 0, 10, 1, 0]);
    expect(Array.from(mesh.faces)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  });

  it("flips winding for a reflected node transform", () => {
    const mesh = parseGlb(makeGlb({ nodeScale: [-1, 1, 1] }));
    expect(Array.from(mesh.faces.slice(0, 3))).toEqual([0, 2, 1]);
    expect(Array.from(mesh.vertices.slice(3, 6))).toEqual([-1, 0, 0]);
  });

  it("applies sparse POSITION accessors with an implicit zero base", () => {
    const mesh = parseGlb(makeSparseGlb());
    expect(Array.from(mesh.vertices)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(mesh.faces)).toEqual([0, 1, 2]);
  });

  it("rejects external buffers instead of silently dropping geometry", () => {
    expect(() => parseGlb(makeGlb({ externalBuffer: true }))).toThrowError(
      expect.objectContaining({ code: "UNSUPPORTED_GLTF", stage: "parsing-input" }),
    );
  });

  it("rejects non-triangle primitives with context", () => {
    try {
      parseGlb(makeGlb({ primitiveMode: 1 }));
      throw new Error("expected parse failure");
    } catch (error) {
      expect(error).toMatchObject({
        code: "UNSUPPORTED_GLTF",
        context: { mesh_index: 0, primitive_index: 0 },
      });
    }
  });

  it("rejects morph targets instead of compiling the wrong shape", () => {
    expect(() => parseGlb(makeGlb({ morphTargets: true }))).toThrowError(
      expect.objectContaining({ code: "UNSUPPORTED_GLTF" }),
    );
  });

  it("rejects malformed container lengths", () => {
    const glb = makeGlb();
    new DataView(glb).setUint32(8, glb.byteLength - 4, true);
    expect(() => parseGlb(glb)).toThrowError(
      expect.objectContaining({ code: "INVALID_GLB" }),
    );
  });

  it("exposes structured error details for UI rendering", () => {
    try {
      parseGlb(new ArrayBuffer(4));
      throw new Error("expected parse failure");
    } catch (error) {
      expect(error).toHaveProperty("toInfo");
      const info = (error as { toInfo(): unknown }).toInfo();
      expect(info).toMatchObject({
        code: "INVALID_GLB",
        stage: "parsing-input",
        retryable: false,
      });
    }
  });
});
