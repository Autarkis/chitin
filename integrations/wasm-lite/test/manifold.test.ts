import { describe, expect, it } from "vitest";

import { analyzeManifold, checkManifold } from "../src/manifold.js";

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
    expect(analyzeManifold(verts, faces)).toMatchObject({
      manifold: false,
      boundary_edge_count: 3,
      non_manifold_edge_count: 0,
      degenerate_triangle_count: 0,
      inconsistent_winding_edge_count: 0,
    });
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

  it("rejects a zero-area triangle built from distinct vertices", () => {
    // Three collinear points: distinct indices, so the repeated-index check
    // waves it through, but CoACD sees the same zero-area face.
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(/collinear or coincident/);
  });

  it("rejects coincident vertices at distinct indices", () => {
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(/collinear or coincident/);
  });

  it("keeps a thin but real sliver", () => {
    // A sliver 1e-6 of the mesh across is legitimate geometry, not degenerate;
    // the area floor has to sit well below it.
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 0.5, 1e-6, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(/boundary \(open\) edge/);
  });

  it("throws a ChitinError with code NON_MANIFOLD", () => {
    const verts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const faces = new Int32Array([0, 1, 2]);
    expect(() => checkManifold(verts, faces)).toThrow(expect.objectContaining({
      code: "NON_MANIFOLD",
      context: {
        boundary_edges: 3,
        non_manifold_edges: 0,
        degenerate_triangles: 0,
        inconsistent_winding_edges: 0,
      },
    }));
  });

  it("rejects a closed mesh with inconsistent face winding", () => {
    // A consistently-wound tetrahedron, with one face's winding reversed.
    // Every undirected edge is still shared by exactly two triangles -- the
    // mesh looks closed and manifold under an undirected edge count -- but
    // the flipped face traverses its three edges in the same direction as
    // their partner triangle instead of the opposite one. That is exactly
    // the pattern a mirrored modifier or boolean export leaves behind, and
    // it makes signed-volume computations (e.g. enclosedVolume) cancel
    // toward zero even though the mesh is topologically closed.
    const verts = new Float64Array([
      0, 0, 0, // A
      1, 0, 0, // B
      0, 1, 0, // C
      0, 0, 1, // D
    ]);
    const consistent = new Int32Array([
      0, 2, 1, // A, C, B (opposite D)
      0, 1, 3, // A, B, D (opposite C)
      0, 3, 2, // A, D, C (opposite B)
      1, 2, 3, // B, C, D (opposite A)
    ]);
    expect(() => checkManifold(verts, consistent)).not.toThrow();

    const flipped = new Int32Array([
      0, 2, 1,
      0, 1, 3,
      0, 3, 2,
      1, 3, 2, // reversed: B, D, C
    ]);
    expect(() => checkManifold(verts, flipped)).toThrow(/inconsistent winding/);
    expect(analyzeManifold(verts, flipped)).toMatchObject({
      manifold: false,
      boundary_edge_count: 0,
      non_manifold_edge_count: 0,
      degenerate_triangle_count: 0,
      inconsistent_winding_edge_count: 3,
    });
  });
});
