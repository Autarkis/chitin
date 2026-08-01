import { ChitinError } from "./errors.js";

// CoACD assumes a closed, edge-manifold input; boundary or non-manifold edges
// make it produce garbage hulls or hang. This is an optional O(faces) precheck
// the worker client runs when asked, so callers get a NON_MANIFOLD error up
// front instead of a bad decomposition.

// Squared-area floor for a triangle, relative to the squared bounding
// diagonal. Small enough that a real sliver survives; large enough to catch the
// rounding noise a collinear triple leaves behind instead of an exact zero.
const DEGENERATE_AREA_EPS = 1e-20;

function boundingDiagonalSquared(vertices: Float64Array): number {
  if (vertices.length === 0) return 0;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const v = vertices[i + axis];
      if (v < min[axis]) min[axis] = v;
      if (v > max[axis]) max[axis] = v;
    }
  }
  let sum = 0;
  for (let axis = 0; axis < 3; axis++) {
    const d = max[axis] - min[axis];
    sum += d * d;
  }
  return sum;
}

/** Squared length of the cross product, i.e. (2 * triangle area)^2. */
function doubleAreaSquared(
  vertices: Float64Array,
  i0: number,
  i1: number,
  i2: number,
): number {
  const ax = vertices[i1 * 3] - vertices[i0 * 3];
  const ay = vertices[i1 * 3 + 1] - vertices[i0 * 3 + 1];
  const az = vertices[i1 * 3 + 2] - vertices[i0 * 3 + 2];
  const bx = vertices[i2 * 3] - vertices[i0 * 3];
  const by = vertices[i2 * 3 + 1] - vertices[i0 * 3 + 1];
  const bz = vertices[i2 * 3 + 2] - vertices[i0 * 3 + 2];
  const cx = ay * bz - az * by;
  const cy = az * bx - ax * bz;
  const cz = ax * by - ay * bx;
  return cx * cx + cy * cy + cz * cz;
}

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

  // Three distinct indices can still describe a zero-area triangle -- collinear
  // or coincident points -- which CoACD chokes on exactly like a repeated
  // index. Scale the area floor by the mesh's own size so the test means the
  // same thing whether the asset is in millimetres or metres, and keep it tight
  // enough that a genuinely thin sliver is not mistaken for a degenerate one.
  const diag2 = boundingDiagonalSquared(vertices);
  const minDoubleAreaSquared = DEGENERATE_AREA_EPS * diag2 * diag2;

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
    if (doubleAreaSquared(vertices, i0, i1, i2) <= minDoubleAreaSquared) {
      throw new ChitinError(
        "NON_MANIFOLD",
        `degenerate triangle at face ${f / 3}: zero area from distinct vertices ` +
          `(${i0}, ${i1}, ${i2}) -- collinear or coincident points`,
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
