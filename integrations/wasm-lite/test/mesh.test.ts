import { describe, expect, it } from "vitest";

import { canonicalizeMesh, splitMeshComponents } from "../src/mesh.js";

describe("canonicalizeMesh", () => {
  it("welds exact render seams and removes duplicate faces", () => {
    const vertices = new Float64Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0,
    ]);
    const mesh = canonicalizeMesh(vertices, new Int32Array([0, 1, 2, 3, 4, 5]));

    expect(Array.from(mesh.vertices)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(mesh.faces)).toEqual([0, 1, 2]);
    expect(mesh).toMatchObject({
      source_vertex_count: 6,
      welded_vertex_count: 3,
      removed_degenerate_triangles: 0,
      removed_duplicate_triangles: 1,
    });
  });

  it("drops zero-area triangles and unused vertices", () => {
    const vertices = new Float64Array([
      0, 0, 0, 1, 0, 0, 2, 0, 0,
      0, 1, 0, 9, 9, 9,
    ]);
    const mesh = canonicalizeMesh(vertices, new Int32Array([0, 1, 2, 0, 1, 3]));

    expect(Array.from(mesh.faces)).toEqual([0, 1, 2]);
    expect(Array.from(mesh.vertices)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(mesh.removed_degenerate_triangles).toBe(1);
    expect(mesh.welded_vertex_count).toBe(3);
  });

  it("rejects a mesh emptied by canonicalization", () => {
    expect(() => canonicalizeMesh(
      new Float64Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
      new Int32Array([0, 1, 2]),
    )).toThrow(/no non-degenerate triangles/);
  });
});

describe("splitMeshComponents", () => {
  it("returns compact components in first-face order", () => {
    const vertices = new Float64Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0,
      10, 0, 0, 11, 0, 0, 10, 1, 0,
    ]);
    const components = splitMeshComponents(vertices, new Int32Array([3, 4, 5, 0, 1, 2]));

    expect(components).toHaveLength(2);
    expect(Array.from(components[0].vertices)).toEqual([10, 0, 0, 11, 0, 0, 10, 1, 0]);
    expect(Array.from(components[0].faces)).toEqual([0, 1, 2]);
    expect(Array.from(components[1].vertices)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  });
});
