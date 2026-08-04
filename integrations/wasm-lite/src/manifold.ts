import { ChitinError } from "./errors.js";

// CoACD assumes a closed, edge-manifold input; boundary or non-manifold edges
// make the lightweight WASM build abort. The scene compiler runs this O(faces)
// precheck by default so callers get a contextual NON_MANIFOLD error first.

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

export interface ManifoldAnalysis {
  boundary_edge_count: number;
  non_manifold_edge_count: number;
  degenerate_triangle_count: number;
  first_problem: string | null;
  manifold: boolean;
}

/** Measure every topology failure instead of stopping at the first bad edge. */
export function analyzeManifold(
  vertices: Float64Array,
  faces: Int32Array,
): ManifoldAnalysis {
  const vertexCount = vertices.length / 3;
  const edgeUses = new Map<number, number>();
  const diag2 = boundingDiagonalSquared(vertices);
  const minDoubleAreaSquared = DEGENERATE_AREA_EPS * diag2 * diag2;
  let degenerateTriangleCount = 0;
  let firstProblem: string | null = null;

  for (let f = 0; f < faces.length; f += 3) {
    const i0 = faces[f];
    const i1 = faces[f + 1];
    const i2 = faces[f + 2];
    let degenerate: string | null = null;
    if (i0 === i1 || i1 === i2 || i0 === i2) {
      degenerate = `degenerate triangle at face ${f / 3}: repeats a vertex (${i0}, ${i1}, ${i2})`;
    } else if (doubleAreaSquared(vertices, i0, i1, i2) <= minDoubleAreaSquared) {
      degenerate =
        `degenerate triangle at face ${f / 3}: zero area from distinct vertices ` +
        `(${i0}, ${i1}, ${i2}) -- collinear or coincident points`;
    }
    if (degenerate) {
      degenerateTriangleCount++;
      firstProblem ??= degenerate;
      continue;
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

  let boundaryEdgeCount = 0;
  let nonManifoldEdgeCount = 0;
  for (const [key, uses] of edgeUses) {
    if (uses === 2) continue;
    const a = Math.floor(key / vertexCount);
    const b = key % vertexCount;
    if (uses === 1) {
      boundaryEdgeCount++;
      firstProblem ??= `boundary (open) edge between vertices ${a} and ${b}`;
    } else {
      nonManifoldEdgeCount++;
      firstProblem ??= `non-manifold edge (${uses} triangles) between vertices ${a} and ${b}`;
    }
  }

  return {
    boundary_edge_count: boundaryEdgeCount,
    non_manifold_edge_count: nonManifoldEdgeCount,
    degenerate_triangle_count: degenerateTriangleCount,
    first_problem: firstProblem,
    manifold:
      boundaryEdgeCount === 0 &&
      nonManifoldEdgeCount === 0 &&
      degenerateTriangleCount === 0,
  };
}

/**
 * Verify that the triangle mesh is edge-manifold and closed: every undirected
 * edge is shared by exactly two triangles, and no triangle is degenerate.
 * Throws {@link ChitinError} with code `NON_MANIFOLD` and aggregate counts.
 *
 * Assumes the shape/index-range checks from `validateMeshInput` already hold;
 * it re-derives only what it needs to key edges.
 */
export function checkManifold(vertices: Float64Array, faces: Int32Array): void {
  const analysis = analyzeManifold(vertices, faces);
  if (analysis.manifold) return;
  const counts = [
    `${analysis.boundary_edge_count} boundary edge${analysis.boundary_edge_count === 1 ? "" : "s"}`,
    `${analysis.non_manifold_edge_count} non-manifold edge${analysis.non_manifold_edge_count === 1 ? "" : "s"}`,
    `${analysis.degenerate_triangle_count} degenerate triangle${analysis.degenerate_triangle_count === 1 ? "" : "s"}`,
  ].join(", ");
  throw new ChitinError(
    "NON_MANIFOLD",
    `${analysis.first_problem}; topology summary: ${counts}`,
    {
      context: {
        boundary_edges: analysis.boundary_edge_count,
        non_manifold_edges: analysis.non_manifold_edge_count,
        degenerate_triangles: analysis.degenerate_triangle_count,
      },
    },
  );
}
