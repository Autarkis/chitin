import type { TriangleMesh } from "./mesh.js";

export interface GaussianFieldInput {
  centers: ArrayLike<number>;
  scales: ArrayLike<number>;
  quaternions: ArrayLike<number>;
  opacities?: ArrayLike<number>;
}

export interface GaussianFieldReconstructionOptions {
  resolution?: number;
  supportMultiplier?: number;
  minOpacity?: number;
  shellThickness?: number;
  minComponentVoxels?: number;
}

export interface SplatPreprocessResult {
  positions: Float64Array;
  normals: Float64Array;
}

export interface ProximityFilterOptions {
  maxDistanceRatio: number;
}

export interface ThinShellOptions {
  thickness: number;
}

/**
 * Derive surface normals from Gaussian covariance parameters.
 *
 * Each Gaussian's minor scale axis (the thinnest direction of the ellipsoid)
 * points approximately perpendicular to the captured surface. This extracts
 * that axis from the quaternion rotation matrix and normalizes it.
 */
export function normalsFromCovariance(
  scales: ArrayLike<number>,
  quaternions: ArrayLike<number>,
  count: number,
  logScale = true,
): Float64Array {
  const normals = new Float64Array(count * 3);
  for (let i = 0; i < count; i++) {
    const si = i * 3;
    let sx = scales[si];
    let sy = scales[si + 1];
    let sz = scales[si + 2];
    if (logScale) {
      sx = Math.exp(sx);
      sy = Math.exp(sy);
      sz = Math.exp(sz);
    }

    const qi = i * 4;
    let qx = quaternions[qi];
    let qy = quaternions[qi + 1];
    let qz = quaternions[qi + 2];
    let qw = quaternions[qi + 3];
    const qnorm = Math.hypot(qx, qy, qz, qw) || 1;
    qx /= qnorm;
    qy /= qnorm;
    qz /= qnorm;
    qw /= qnorm;

    // Rotation matrix columns (R[:, col]) for the minor-scale axis
    let minAxis: number;
    if (sx <= sy && sx <= sz) minAxis = 0;
    else if (sy <= sz) minAxis = 1;
    else minAxis = 2;

    let nx: number, ny: number, nz: number;
    if (minAxis === 0) {
      nx = 1 - 2 * (qy * qy + qz * qz);
      ny = 2 * (qx * qy + qw * qz);
      nz = 2 * (qx * qz - qw * qy);
    } else if (minAxis === 1) {
      nx = 2 * (qx * qy - qw * qz);
      ny = 1 - 2 * (qx * qx + qz * qz);
      nz = 2 * (qy * qz + qw * qx);
    } else {
      nx = 2 * (qx * qz + qw * qy);
      ny = 2 * (qy * qz - qw * qx);
      nz = 1 - 2 * (qx * qx + qy * qy);
    }

    const len = Math.hypot(nx, ny, nz) || 1;
    normals[si] = nx / len;
    normals[si + 1] = ny / len;
    normals[si + 2] = nz / len;
  }
  return normals;
}

/**
 * Sign-correct normals so they consistently point outward.
 *
 * Uses a centroid-based heuristic: for each normal, if it points away from
 * the centroid of all positions it keeps its sign; otherwise it flips.
 * This is simpler than Hoppe's MST propagation (which Python chitin uses
 * via Open3D) but works well for convex-ish objects — the typical case for
 * individual splat captures.
 */
export function orientNormalsConsistently(
  centers: ArrayLike<number>,
  normals: Float64Array,
  count: number,
): Float64Array {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < count; i++) {
    cx += centers[i * 3];
    cy += centers[i * 3 + 1];
    cz += centers[i * 3 + 2];
  }
  cx /= count;
  cy /= count;
  cz /= count;

  const oriented = new Float64Array(normals);
  for (let i = 0; i < count; i++) {
    const oi = i * 3;
    const dx = centers[oi] - cx;
    const dy = centers[oi + 1] - cy;
    const dz = centers[oi + 2] - cz;
    const dot = oriented[oi] * dx + oriented[oi + 1] * dy + oriented[oi + 2] * dz;
    if (dot < 0) {
      oriented[oi] = -oriented[oi];
      oriented[oi + 1] = -oriented[oi + 1];
      oriented[oi + 2] = -oriented[oi + 2];
    }
  }
  return oriented;
}

/**
 * Densify the point cloud by sampling 4 extra points per Gaussian along
 * its two major ellipsoid axes, matching Python chitin's inflate_splat_points.
 *
 * Returns positions of length (count * 5 * 3) — original center plus
 * ±major and ±minor axis offsets scaled by surfaceRatio.
 */
export function inflateSplatPoints(
  centers: ArrayLike<number>,
  scales: ArrayLike<number>,
  quaternions: ArrayLike<number>,
  count: number,
  surfaceRatio: number,
  logScale = true,
): Float64Array {
  const inflated = new Float64Array(count * 5 * 3);

  for (let i = 0; i < count; i++) {
    const ci = i * 3;
    const qi = i * 4;
    const px = centers[ci], py = centers[ci + 1], pz = centers[ci + 2];

    let sx = scales[ci], sy = scales[ci + 1], sz = scales[ci + 2];
    if (logScale) {
      sx = Math.exp(sx);
      sy = Math.exp(sy);
      sz = Math.exp(sz);
    }

    let qx = quaternions[qi], qy = quaternions[qi + 1];
    let qz = quaternions[qi + 2], qw = quaternions[qi + 3];
    const qnorm = Math.hypot(qx, qy, qz, qw) || 1;
    qx /= qnorm; qy /= qnorm; qz /= qnorm; qw /= qnorm;

    // Sort axes by scale to find major (largest) and minor (middle)
    const scaleArr = [sx, sy, sz];
    const sorted = [0, 1, 2].sort((a, b) => scaleArr[a] - scaleArr[b]);
    const majorIdx = sorted[2];
    const minorIdx = sorted[1];

    // Rotation matrix column for an axis
    const rotCol = (col: number): [number, number, number] => {
      if (col === 0) return [
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qw * qz),
        2 * (qx * qz - qw * qy),
      ];
      if (col === 1) return [
        2 * (qx * qy - qw * qz),
        1 - 2 * (qx * qx + qz * qz),
        2 * (qy * qz + qw * qx),
      ];
      return [
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy),
      ];
    };

    const [ax, ay, az] = rotCol(majorIdx);
    const majorScale = scaleArr[majorIdx] * surfaceRatio;
    const dax = ax * majorScale, day = ay * majorScale, daz = az * majorScale;

    const [bx, by, bz] = rotCol(minorIdx);
    const minorScale = scaleArr[minorIdx] * surfaceRatio;
    const dbx = bx * minorScale, dby = by * minorScale, dbz = bz * minorScale;

    const base = i * 5 * 3;
    // Original center
    inflated[base] = px; inflated[base + 1] = py; inflated[base + 2] = pz;
    // +major
    inflated[base + 3] = px + dax; inflated[base + 4] = py + day; inflated[base + 5] = pz + daz;
    // -major
    inflated[base + 6] = px - dax; inflated[base + 7] = py - day; inflated[base + 8] = pz - daz;
    // +minor
    inflated[base + 9] = px + dbx; inflated[base + 10] = py + dby; inflated[base + 11] = pz + dbz;
    // -minor
    inflated[base + 12] = px - dbx; inflated[base + 13] = py - dby; inflated[base + 14] = pz - dbz;
  }
  return inflated;
}

/**
 * Expand normals to match inflated points: each original normal is repeated
 * 5 times (center + 4 offset copies).
 */
export function tileNormals(normals: Float64Array, count: number): Float64Array {
  const tiled = new Float64Array(count * 5 * 3);
  for (let i = 0; i < count; i++) {
    const si = i * 3;
    const nx = normals[si], ny = normals[si + 1], nz = normals[si + 2];
    for (let s = 0; s < 5; s++) {
      const ti = (i * 5 + s) * 3;
      tiled[ti] = nx; tiled[ti + 1] = ny; tiled[ti + 2] = nz;
    }
  }
  return tiled;
}

/**
 * Run the full preprocessing pipeline on a Gaussian field to produce the
 * oriented, inflated point cloud ready for Poisson reconstruction.
 */
export function preprocessGaussianField(
  input: GaussianFieldInput,
  options: { surfaceRatio?: number; minOpacity?: number; logScale?: boolean } = {},
): SplatPreprocessResult {
  const surfaceRatio = options.surfaceRatio ?? 0.5;
  const minOpacity = options.minOpacity ?? 0.2;
  const logScale = options.logScale ?? false;
  const totalCount = input.centers.length / 3;

  // Filter by opacity
  const activeIndices: number[] = [];
  for (let i = 0; i < totalCount; i++) {
    if ((input.opacities?.[i] ?? 1) >= minOpacity) activeIndices.push(i);
  }
  const count = activeIndices.length;
  if (count === 0) throw new Error("No Gaussians meet minOpacity threshold");

  const centers = new Float64Array(count * 3);
  const scales = new Float64Array(count * 3);
  const quaternions = new Float64Array(count * 4);
  for (let i = 0; i < count; i++) {
    const src = activeIndices[i];
    centers[i * 3] = input.centers[src * 3];
    centers[i * 3 + 1] = input.centers[src * 3 + 1];
    centers[i * 3 + 2] = input.centers[src * 3 + 2];
    scales[i * 3] = input.scales[src * 3];
    scales[i * 3 + 1] = input.scales[src * 3 + 1];
    scales[i * 3 + 2] = input.scales[src * 3 + 2];
    quaternions[i * 4] = input.quaternions[src * 4];
    quaternions[i * 4 + 1] = input.quaternions[src * 4 + 1];
    quaternions[i * 4 + 2] = input.quaternions[src * 4 + 2];
    quaternions[i * 4 + 3] = input.quaternions[src * 4 + 3];
  }

  const rawNormals = normalsFromCovariance(scales, quaternions, count, logScale);
  const normals = orientNormalsConsistently(centers, rawNormals, count);
  const inflatedPositions = inflateSplatPoints(
    centers, scales, quaternions, count, surfaceRatio, logScale,
  );
  const inflatedNormals = tileNormals(normals, count);

  return { positions: inflatedPositions, normals: inflatedNormals };
}

/**
 * Remove mesh vertices whose nearest input point exceeds a distance threshold.
 *
 * Uses a grid-based spatial hash instead of scipy's cKDTree. The threshold
 * is maxDistanceRatio × median nearest-neighbor distance in the input cloud.
 */
export function proximityFilterMesh(
  meshVerts: Float64Array,
  meshFaces: Int32Array,
  inputPositions: Float64Array,
  maxDistanceRatio: number,
): TriangleMesh {
  const inputCount = inputPositions.length / 3;
  if (inputCount === 0 || meshVerts.length === 0) {
    return { vertices: meshVerts, faces: meshFaces };
  }

  // Estimate characteristic spacing from a sample of input points
  const sampleSize = Math.min(1000, inputCount);
  const cellSize = estimateMedianNN(inputPositions, inputCount, sampleSize);
  const threshold = maxDistanceRatio * cellSize;
  const thresholdSq = threshold * threshold;

  // Build spatial hash of input positions
  const grid = buildSpatialHash(inputPositions, inputCount, threshold);

  // Test each mesh vertex
  const meshVertCount = meshVerts.length / 3;
  const keep = new Uint8Array(meshVertCount);
  for (let i = 0; i < meshVertCount; i++) {
    const x = meshVerts[i * 3];
    const y = meshVerts[i * 3 + 1];
    const z = meshVerts[i * 3 + 2];
    if (hasNearbyPoint(grid, inputPositions, x, y, z, threshold, thresholdSq)) {
      keep[i] = 1;
    }
  }

  // Remap vertices and filter faces
  const oldToNew = new Int32Array(meshVertCount).fill(-1);
  let newCount = 0;
  for (let i = 0; i < meshVertCount; i++) {
    if (keep[i]) oldToNew[i] = newCount++;
  }
  const newVerts = new Float64Array(newCount * 3);
  for (let i = 0; i < meshVertCount; i++) {
    if (keep[i]) {
      const ni = oldToNew[i] * 3;
      newVerts[ni] = meshVerts[i * 3];
      newVerts[ni + 1] = meshVerts[i * 3 + 1];
      newVerts[ni + 2] = meshVerts[i * 3 + 2];
    }
  }

  const faceCount = meshFaces.length / 3;
  const newFaceList: number[] = [];
  for (let f = 0; f < faceCount; f++) {
    const a = oldToNew[meshFaces[f * 3]];
    const b = oldToNew[meshFaces[f * 3 + 1]];
    const c = oldToNew[meshFaces[f * 3 + 2]];
    if (a >= 0 && b >= 0 && c >= 0) {
      newFaceList.push(a, b, c);
    }
  }

  return {
    vertices: newVerts,
    faces: new Int32Array(newFaceList),
  };
}

/**
 * Double a mesh into outer + inner shells offset along vertex normals,
 * stitching boundary edges to produce a watertight solid.
 */
export function extrudeThinShell(
  vertices: Float64Array,
  faces: Int32Array,
  thickness: number,
): TriangleMesh {
  const n = vertices.length / 3;
  const faceCount = faces.length / 3;
  const vnormals = computeVertexNormals(vertices, faces);

  const half = thickness / 2;
  const allVerts = new Float64Array(n * 2 * 3);
  for (let i = 0; i < n; i++) {
    const vi = i * 3;
    allVerts[vi] = vertices[vi] + vnormals[vi] * half;
    allVerts[vi + 1] = vertices[vi + 1] + vnormals[vi + 1] * half;
    allVerts[vi + 2] = vertices[vi + 2] + vnormals[vi + 2] * half;
    const ii = (i + n) * 3;
    allVerts[ii] = vertices[vi] - vnormals[vi] * half;
    allVerts[ii + 1] = vertices[vi + 1] - vnormals[vi + 1] * half;
    allVerts[ii + 2] = vertices[vi + 2] - vnormals[vi + 2] * half;
  }

  // Outer faces (original winding) + inner faces (reversed winding, offset by n)
  const allFacesList: number[] = [];
  for (let f = 0; f < faceCount; f++) {
    const fi = f * 3;
    allFacesList.push(faces[fi], faces[fi + 1], faces[fi + 2]);
  }
  for (let f = 0; f < faceCount; f++) {
    const fi = f * 3;
    allFacesList.push(faces[fi + 2] + n, faces[fi + 1] + n, faces[fi] + n);
  }

  // Find boundary edges and stitch
  const edgeCounts = new Map<number, number>();
  const edgeKey = (a: number, b: number): number =>
    Math.min(a, b) * (n + 1) + Math.max(a, b);

  for (let f = 0; f < faceCount; f++) {
    const fi = f * 3;
    const pairs = [
      [faces[fi], faces[fi + 1]],
      [faces[fi + 1], faces[fi + 2]],
      [faces[fi + 2], faces[fi]],
    ] as const;
    for (const [a, b] of pairs) {
      const key = edgeKey(a, b);
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }

  const boundaryKeys = new Set<number>();
  for (const [key, count] of edgeCounts) {
    if (count === 1) boundaryKeys.add(key);
  }

  for (let f = 0; f < faceCount; f++) {
    const fi = f * 3;
    const pairs = [
      [faces[fi], faces[fi + 1]],
      [faces[fi + 1], faces[fi + 2]],
      [faces[fi + 2], faces[fi]],
    ] as const;
    for (const [e0, e1] of pairs) {
      if (boundaryKeys.has(edgeKey(e0, e1))) {
        allFacesList.push(e0, e1, e1 + n);
        allFacesList.push(e0, e1 + n, e0 + n);
      }
    }
  }

  return {
    vertices: allVerts,
    faces: new Int32Array(allFacesList),
  };
}

function computeVertexNormals(
  vertices: Float64Array,
  faces: Int32Array,
): Float64Array {
  const n = vertices.length / 3;
  const faceCount = faces.length / 3;
  const vnormals = new Float64Array(n * 3);

  for (let f = 0; f < faceCount; f++) {
    const fi = f * 3;
    const i0 = faces[fi] * 3, i1 = faces[fi + 1] * 3, i2 = faces[fi + 2] * 3;
    const e1x = vertices[i1] - vertices[i0];
    const e1y = vertices[i1 + 1] - vertices[i0 + 1];
    const e1z = vertices[i1 + 2] - vertices[i0 + 2];
    const e2x = vertices[i2] - vertices[i0];
    const e2y = vertices[i2 + 1] - vertices[i0 + 1];
    const e2z = vertices[i2 + 2] - vertices[i0 + 2];
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    vnormals[i0] += nx; vnormals[i0 + 1] += ny; vnormals[i0 + 2] += nz;
    vnormals[i1] += nx; vnormals[i1 + 1] += ny; vnormals[i1 + 2] += nz;
    vnormals[i2] += nx; vnormals[i2 + 1] += ny; vnormals[i2 + 2] += nz;
  }

  for (let i = 0; i < n; i++) {
    const vi = i * 3;
    const len = Math.hypot(vnormals[vi], vnormals[vi + 1], vnormals[vi + 2]) || 1;
    vnormals[vi] /= len;
    vnormals[vi + 1] /= len;
    vnormals[vi + 2] /= len;
  }
  return vnormals;
}

// --- Spatial hash helpers for proximity filtering ---

interface SpatialHash {
  cells: Map<number, number[]>;
  cellSize: number;
  originX: number;
  originY: number;
  originZ: number;
}

function buildSpatialHash(
  positions: Float64Array,
  count: number,
  cellSize: number,
): SpatialHash {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
  }

  const cells = new Map<number, number[]>();
  const invCell = 1 / cellSize;
  const W = 16384;

  for (let i = 0; i < count; i++) {
    const gx = Math.floor((positions[i * 3] - minX) * invCell);
    const gy = Math.floor((positions[i * 3 + 1] - minY) * invCell);
    const gz = Math.floor((positions[i * 3 + 2] - minZ) * invCell);
    const key = gx + gy * W + gz * W * W;
    let cell = cells.get(key);
    if (cell === undefined) {
      cell = [];
      cells.set(key, cell);
    }
    cell.push(i);
  }

  return { cells, cellSize, originX: minX, originY: minY, originZ: minZ };
}

function hasNearbyPoint(
  grid: SpatialHash,
  positions: Float64Array,
  x: number,
  y: number,
  z: number,
  threshold: number,
  thresholdSq: number,
): boolean {
  const invCell = 1 / grid.cellSize;
  const W = 16384;
  const gx = Math.floor((x - grid.originX) * invCell);
  const gy = Math.floor((y - grid.originY) * invCell);
  const gz = Math.floor((z - grid.originZ) * invCell);

  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const key = (gx + dx) + (gy + dy) * W + (gz + dz) * W * W;
        const cell = grid.cells.get(key);
        if (cell === undefined) continue;
        for (const idx of cell) {
          const pi = idx * 3;
          const ddx = positions[pi] - x;
          const ddy = positions[pi + 1] - y;
          const ddz = positions[pi + 2] - z;
          if (ddx * ddx + ddy * ddy + ddz * ddz <= thresholdSq) return true;
        }
      }
    }
  }
  return false;
}

function estimateMedianNN(
  positions: Float64Array,
  count: number,
  sampleSize: number,
): number {
  const grid = buildSpatialHash(positions, count, 1.0);
  const distances: number[] = [];
  const step = Math.max(1, Math.floor(count / sampleSize));

  for (let i = 0; i < count && distances.length < sampleSize; i += step) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    let bestDist = Infinity;
    const invCell = 1 / grid.cellSize;
    const W = 16384;
    const gx = Math.floor((x - grid.originX) * invCell);
    const gy = Math.floor((y - grid.originY) * invCell);
    const gz = Math.floor((z - grid.originZ) * invCell);

    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const key = (gx + dx) + (gy + dy) * W + (gz + dz) * W * W;
          const cell = grid.cells.get(key);
          if (cell === undefined) continue;
          for (const idx of cell) {
            if (idx === i) continue;
            const pi = idx * 3;
            const ddx = positions[pi] - x;
            const ddy = positions[pi + 1] - y;
            const ddz = positions[pi + 2] - z;
            const dist = ddx * ddx + ddy * ddy + ddz * ddz;
            if (dist < bestDist) bestDist = dist;
          }
        }
      }
    }
    if (bestDist < Infinity) distances.push(Math.sqrt(bestDist));
  }

  if (distances.length === 0) return 1;
  distances.sort((a, b) => a - b);
  return distances[Math.floor(distances.length / 2)];
}
