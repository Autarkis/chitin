import { ChitinError } from "./errors.js";
import { meshBounds, triangleCrossProduct, vertexCount } from "./geometry.js";

// CoACD assumes a closed, edge-manifold input; boundary or non-manifold edges
// make the lightweight WASM build abort. The scene compiler runs this O(faces)
// precheck by default so callers get a contextual NON_MANIFOLD error first.

// Squared-area floor for a triangle, relative to the squared bounding
// diagonal. Small enough that a real sliver survives; large enough to catch the
// rounding noise a collinear triple leaves behind instead of an exact zero.
const DEGENERATE_AREA_EPS = 1e-20;

function boundingDiagonalSquared(vertices: Float64Array): number {
  if (vertices.length === 0) return 0;
  const { min, max } = meshBounds(vertices);
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
  const [cx, cy, cz] = triangleCrossProduct(vertices, i0, i1, i2);
  return cx * cx + cy * cy + cz * cz;
}

export interface ManifoldAnalysis {
  boundary_edge_count: number;
  non_manifold_edge_count: number;
  degenerate_triangle_count: number;
  inconsistent_winding_edge_count: number;
  first_problem: string | null;
  manifold: boolean;
}

/** Measure every topology failure instead of stopping at the first bad edge. */
export function analyzeManifold(
  vertices: Float64Array,
  faces: Int32Array,
): ManifoldAnalysis {
  const vertexTotal = vertexCount(vertices);
  const edgeUses = new Map<number, number>();
  // Signed accumulator per undirected edge key: +1 for each half-edge
  // traversed a < b, -1 for each traversed a > b. A consistently wound
  // closed manifold visits every edge once in each direction, so the two
  // half-edges cancel to 0. If both triangles traverse it the same way
  // (both +1 or both -1), the sum is +/-2 and the winding is inconsistent.
  const edgeDirections = new Map<number, number>();
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
      const key = a < b ? a * vertexTotal + b : b * vertexTotal + a;
      edgeUses.set(key, (edgeUses.get(key) ?? 0) + 1);
      edgeDirections.set(key, (edgeDirections.get(key) ?? 0) + (a < b ? 1 : -1));
    }
  }

  let boundaryEdgeCount = 0;
  let nonManifoldEdgeCount = 0;
  let inconsistentWindingEdgeCount = 0;
  for (const [key, uses] of edgeUses) {
    const a = Math.floor(key / vertexTotal);
    const b = key % vertexTotal;
    if (uses === 1) {
      boundaryEdgeCount++;
      firstProblem ??= `boundary (open) edge between vertices ${a} and ${b}`;
    } else if (uses === 2) {
      if (Math.abs(edgeDirections.get(key) ?? 0) === 2) {
        inconsistentWindingEdgeCount++;
        firstProblem ??=
          `inconsistent winding at edge between vertices ${a} and ${b}: ` +
          `both triangles traverse it in the same direction`;
      }
    } else {
      nonManifoldEdgeCount++;
      firstProblem ??= `non-manifold edge (${uses} triangles) between vertices ${a} and ${b}`;
    }
  }

  return {
    boundary_edge_count: boundaryEdgeCount,
    non_manifold_edge_count: nonManifoldEdgeCount,
    degenerate_triangle_count: degenerateTriangleCount,
    inconsistent_winding_edge_count: inconsistentWindingEdgeCount,
    first_problem: firstProblem,
    manifold:
      boundaryEdgeCount === 0 &&
      nonManifoldEdgeCount === 0 &&
      degenerateTriangleCount === 0 &&
      inconsistentWindingEdgeCount === 0,
  };
}

/**
 * Verify that the triangle mesh is edge-manifold, closed, and consistently
 * wound: every undirected edge is shared by exactly two triangles, one
 * traversing it in each direction, and no triangle is degenerate. A mesh
 * where each edge is used exactly twice but both uses run the same direction
 * (e.g. a mirrored modifier or boolean export that left some faces flipped)
 * passes the undirected edge-count check yet is not consistently wound;
 * `inconsistent_winding_edge_count` catches that case.
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
    `${analysis.inconsistent_winding_edge_count} inconsistent-winding edge${analysis.inconsistent_winding_edge_count === 1 ? "" : "s"}`,
  ].join(", ");
  throw new ChitinError(
    "NON_MANIFOLD",
    `${analysis.first_problem}; topology summary: ${counts}`,
    {
      context: {
        boundary_edges: analysis.boundary_edge_count,
        non_manifold_edges: analysis.non_manifold_edge_count,
        degenerate_triangles: analysis.degenerate_triangle_count,
        inconsistent_winding_edges: analysis.inconsistent_winding_edge_count,
      },
    },
  );
}
