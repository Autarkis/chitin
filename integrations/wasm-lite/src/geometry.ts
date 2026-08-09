import type { TriangleMesh } from "./mesh.js";

export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
}

export type Point = [number, number, number];

export function meshBounds(vertices: ArrayLike<number>): Bounds {
  const min: Point = [Infinity, Infinity, Infinity];
  const max: Point = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < vertices.length; offset += 3) {
    min[0] = Math.min(min[0], vertices[offset]);
    min[1] = Math.min(min[1], vertices[offset + 1]);
    min[2] = Math.min(min[2], vertices[offset + 2]);
    max[0] = Math.max(max[0], vertices[offset]);
    max[1] = Math.max(max[1], vertices[offset + 1]);
    max[2] = Math.max(max[2], vertices[offset + 2]);
  }
  return { min, max };
}

export function boundsDiagonal(bounds: Bounds): number {
  return Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  );
}

export function boundsVolume(bounds: Bounds): number {
  return Math.max(0, bounds.max[0] - bounds.min[0]) *
    Math.max(0, bounds.max[1] - bounds.min[1]) *
    Math.max(0, bounds.max[2] - bounds.min[2]);
}

export function containsBounds(bounds: Bounds, point: Point, tolerance: number): boolean {
  return point[0] >= bounds.min[0] - tolerance && point[0] <= bounds.max[0] + tolerance &&
    point[1] >= bounds.min[1] - tolerance && point[1] <= bounds.max[1] + tolerance &&
    point[2] >= bounds.min[2] - tolerance && point[2] <= bounds.max[2] + tolerance;
}

/** Cross product of two triangle edges in a flat xyz array. */
export function triangleCrossProduct(
  vertices: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
): [number, number, number] {
  const ai = a * 3;
  const bi = b * 3;
  const ci = c * 3;
  const abx = vertices[bi] - vertices[ai];
  const aby = vertices[bi + 1] - vertices[ai + 1];
  const abz = vertices[bi + 2] - vertices[ai + 2];
  const acx = vertices[ci] - vertices[ai];
  const acy = vertices[ci + 1] - vertices[ai + 1];
  const acz = vertices[ci + 2] - vertices[ai + 2];
  return [
    aby * acz - abz * acy,
    abz * acx - abx * acz,
    abx * acy - aby * acx,
  ];
}

export function triangleArea(vertices: ArrayLike<number>, a: number, b: number, c: number): number {
  const [cx, cy, cz] = triangleCrossProduct(vertices, a, b, c);
  return 0.5 * Math.hypot(cx, cy, cz);
}

export function vertexCount(positions: ArrayLike<number>): number {
  return positions.length / 3;
}

export function triangleCount(indices: ArrayLike<number>): number {
  return indices.length / 3;
}

export function componentArea(mesh: TriangleMesh): number {
  let area = 0;
  for (let offset = 0; offset < mesh.faces.length; offset += 3) {
    area += triangleArea(
      mesh.vertices,
      mesh.faces[offset],
      mesh.faces[offset + 1],
      mesh.faces[offset + 2],
    );
  }
  return area;
}
