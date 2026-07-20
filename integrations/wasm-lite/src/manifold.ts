import { ChitinError } from "./errors.js";

// CoACD assumes a closed, edge-manifold input; boundary or non-manifold edges
// make it produce garbage hulls or hang. This is an optional O(faces) precheck
// the worker client runs when asked, so callers get a NON_MANIFOLD error up
// front instead of a bad decomposition.

/**
 * Verify that the triangle mesh is edge-manifold and closed: every undirected
 * edge is shared by exactly two triangles, and no triangle is degenerate.
 * Throws {@link ChitinError} with code `NON_MANIFOLD` on the first violation.
 *
 * Assumes the shape/index-range checks from `validateMeshInput` already hold;
 * it re-derives only what it needs to key edges.
 */
export function checkManifold(vertices: Float64Array, faces: Int32Array): void {
  const vertexCount = vertices.length / 3;
  // Encode an undirected edge (a, b) as a single integer key with a < b. This
  // stays exact as long as vertexCount^2 < 2^53, which holds for any real mesh.
  const edgeUses = new Map<number, number>();

  for (let f = 0; f < faces.length; f += 3) {
    const i0 = faces[f];
    const i1 = faces[f + 1];
    const i2 = faces[f + 2];
    if (i0 === i1 || i1 === i2 || i0 === i2) {
      throw new ChitinError(
        "NON_MANIFOLD",
        `degenerate triangle at face ${f / 3}: repeats a vertex (${i0}, ${i1}, ${i2})`,
      );
    }
    for (const [a, b] of [
      [i0, i1],
      [i1, i2],
      [i2, i0],
    ]) {
      const key = a < b ? a * vertexCount + b : b * vertexCount + a;
      edgeUses.set(key, (edgeUses.get(key) ?? 0) + 1);
    }
  }

  for (const [key, uses] of edgeUses) {
    if (uses !== 2) {
      const a = Math.floor(key / vertexCount);
      const b = key % vertexCount;
      const kind = uses === 1 ? "boundary (open) edge" : `non-manifold edge (${uses} triangles)`;
      throw new ChitinError(
        "NON_MANIFOLD",
        `${kind} between vertices ${a} and ${b}`,
      );
    }
  }
}
