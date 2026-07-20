import { describe, expect, it } from "vitest";

import { checkManifold } from "../src/manifold.js";

// A closed unit cube: 8 corners, 12 triangles. Every undirected edge is shared
// by exactly two triangles.
const CUBE_VERTS = new Float64Array([
  0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, // z = 0
  0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, // z = 1
]);
const CUBE_FACES = new Int32Array([
  0, 2, 1, 0, 3, 2, // z = 0
  4, 5, 6, 4, 6, 7, // z = 1
  0, 1, 5, 0, 5, 4, // y = 0
  3, 7, 6, 3, 6, 2, // y = 1
  0, 4, 7, 0, 7, 3, // x = 0
  1, 2, 6, 1, 6, 5, // x = 1
]);

describe("checkManifold", () => {
  it("accepts a closed manifold cube", () => {
    expect(() => checkManifold(CUBE_VERTS, CUBE_FACES)).not.toThrow();
  });

  it("rejects a boundary (open) edge", () => {
    // A lone triangle: all three edges are used once.
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(/boundary \(open\) edge/);
  });

  it("rejects a non-manifold edge shared by three triangles", () => {
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0]);
    const faces = new Int32Array([0, 1, 2, 0, 1, 3, 0, 1, 4]); // edge (0,1) x3
    expect(() => checkManifold(verts, faces)).toThrow(/non-manifold edge \(3 triangles\)/);
  });

  it("rejects a degenerate triangle that repeats a vertex", () => {
    const verts = new Float64Array([0, 0, 0, 1, 0, 0]);
    const faces = new Int32Array([0, 1, 1]);
    expect(() => checkManifold(verts, faces)).toThrow(/degenerate triangle/);
  });

  it("throws a ChitinError with code NON_MANIFOLD", () => {
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(
      expect.objectContaining({ code: "NON_MANIFOLD" }),
    );
  });
});
