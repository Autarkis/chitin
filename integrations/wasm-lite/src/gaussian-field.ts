import { ChitinError } from "./errors.js";
import type { TriangleMesh } from "./mesh.js";
import type { GaussianFieldInput } from "./splat-preprocess.js";

export interface VoxelReconstructionOptions {
  /** Voxels along the longest source axis. Defaults to 72. */
  resolution?: number;
  /** Gaussian support multiplier used during occupancy rasterization. Defaults to 1.75. */
  supportMultiplier?: number;
  /** Ignore Gaussians below this opacity. Defaults to 0.2. */
  minOpacity?: number;
  /** Minimum ellipsoid radius in voxels, used to seal the captured shell. Defaults to 1.15. */
  shellThickness?: number;
  /** Drop disconnected voxel islands smaller than this size. Defaults to 8. */
  minComponentVoxels?: number;
}

interface ResolvedReconstructionOptions {
  resolution: number;
  supportMultiplier: number;
  minOpacity: number;
  shellThickness: number;
  minComponentVoxels: number;
}

interface VoxelField {
  occupied: Uint8Array;
  dimensions: [number, number, number];
  origin: [number, number, number];
  voxelSize: number;
}

const MAX_VOXELS = 8_000_000;

/**
 * Reconstruct a closed triangle surface from oriented Gaussian support, ready
 * for Chitin's normal manifold validation and CoACD decomposition pipeline.
 */
export function reconstructGaussianFieldMesh(
  input: GaussianFieldInput,
  options: VoxelReconstructionOptions = {},
): TriangleMesh {
  const resolved = resolveOptions(options);
  const count = validateInput(input);
  const field = rasterizeGaussianField(input, count, resolved);
  retainVoxelComponents(field, resolved.minComponentVoxels);
  regularizeVoxelTopology(field);
  fillEnclosedVoxels(field);
  return meshVoxelBoundary(field);
}

/** Encode a reconstructed Gaussian field as an in-memory GLB compiler input. */
export function gaussianFieldToGlb(
  input: GaussianFieldInput,
  options: VoxelReconstructionOptions = {},
): ArrayBuffer {
  return encodeTriangleMeshGlb(reconstructGaussianFieldMesh(input, options));
}

function resolveOptions(
  options: VoxelReconstructionOptions,
): ResolvedReconstructionOptions {
  const resolution = options.resolution ?? 72;
  if (!Number.isInteger(resolution) || resolution < 16 || resolution > 160) {
    invalid("resolution must be an integer in [16, 160]");
  }
  const supportMultiplier = options.supportMultiplier ?? 1.75;
  if (!Number.isFinite(supportMultiplier) || supportMultiplier <= 0) {
    invalid("supportMultiplier must be finite and positive");
  }
  const minOpacity = options.minOpacity ?? 0.2;
  if (!Number.isFinite(minOpacity) || minOpacity < 0 || minOpacity > 1) {
    invalid("minOpacity must be between 0 and 1");
  }
  const shellThickness = options.shellThickness ?? 1.15;
  if (!Number.isFinite(shellThickness) || shellThickness < 0.5 || shellThickness > 4) {
    invalid("shellThickness must be between 0.5 and 4 voxels");
  }
  const minComponentVoxels = options.minComponentVoxels ?? 8;
  if (!Number.isInteger(minComponentVoxels) || minComponentVoxels < 1) {
    invalid("minComponentVoxels must be a positive integer");
  }
  return {
    resolution,
    supportMultiplier,
    minOpacity,
    shellThickness,
    minComponentVoxels,
  };
}

function validateInput(input: GaussianFieldInput): number {
  if (input.centers.length === 0 || input.centers.length % 3 !== 0) {
    invalid("centers must contain at least one xyz triplet");
  }
  const count = input.centers.length / 3;
  if (input.scales.length !== count * 3) {
    invalid(`scales must contain ${count * 3} values`);
  }
  if (input.quaternions.length !== count * 4) {
    invalid(`quaternions must contain ${count * 4} values`);
  }
  if (input.opacities !== undefined && input.opacities.length !== count) {
    invalid(`opacities must contain ${count} values`);
  }
  for (let index = 0; index < input.centers.length; index++) {
    if (!Number.isFinite(input.centers[index])) invalid(`center value ${index} is not finite`);
  }
  for (let index = 0; index < input.scales.length; index++) {
    if (!Number.isFinite(input.scales[index]) || input.scales[index] <= 0) {
      invalid(`scale value ${index} must be finite and positive`);
    }
  }
  return count;
}

function rasterizeGaussianField(
  input: GaussianFieldInput,
  count: number,
  options: ResolvedReconstructionOptions,
): VoxelField {
  const boundsMin = [Infinity, Infinity, Infinity];
  const boundsMax = [-Infinity, -Infinity, -Infinity];
  let activeCount = 0;
  for (let index = 0; index < count; index++) {
    if ((input.opacities?.[index] ?? 1) < options.minOpacity) continue;
    activeCount++;
    const centerOffset = index * 3;
    const radius =
      Math.max(
        input.scales[centerOffset],
        input.scales[centerOffset + 1],
        input.scales[centerOffset + 2],
      ) * options.supportMultiplier;
    for (let axis = 0; axis < 3; axis++) {
      const center = input.centers[centerOffset + axis];
      boundsMin[axis] = Math.min(boundsMin[axis], center - radius);
      boundsMax[axis] = Math.max(boundsMax[axis], center + radius);
    }
  }
  if (activeCount === 0) invalid("no Gaussians meet minOpacity");

  const extents = boundsMax.map((value, axis) => value - boundsMin[axis]);
  const longest = Math.max(...extents);
  if (!(longest > 0) || !Number.isFinite(longest)) invalid("Gaussian bounds are degenerate");
  const voxelSize = longest / options.resolution;
  const padding = 2;
  const dimensions = extents.map(
    (extent) => Math.max(3, Math.ceil(extent / voxelSize) + padding * 2),
  ) as [number, number, number];
  const totalVoxels = dimensions[0] * dimensions[1] * dimensions[2];
  if (totalVoxels > MAX_VOXELS) {
    invalid(`reconstruction requires ${totalVoxels} voxels; limit is ${MAX_VOXELS}`);
  }
  const origin = boundsMin.map(
    (value) => value - padding * voxelSize,
  ) as [number, number, number];
  const occupied = new Uint8Array(totalVoxels);

  for (let index = 0; index < count; index++) {
    if ((input.opacities?.[index] ?? 1) < options.minOpacity) continue;
    const centerOffset = index * 3;
    const quaternionOffset = index * 4;
    const cx = input.centers[centerOffset];
    const cy = input.centers[centerOffset + 1];
    const cz = input.centers[centerOffset + 2];
    const sx = Math.max(
      input.scales[centerOffset] * options.supportMultiplier,
      voxelSize * options.shellThickness,
    );
    const sy = Math.max(
      input.scales[centerOffset + 1] * options.supportMultiplier,
      voxelSize * options.shellThickness,
    );
    const sz = Math.max(
      input.scales[centerOffset + 2] * options.supportMultiplier,
      voxelSize * options.shellThickness,
    );
    const radius = Math.max(sx, sy, sz);
    const minX = clamp(Math.floor((cx - radius - origin[0]) / voxelSize), 0, dimensions[0] - 1);
    const maxX = clamp(Math.floor((cx + radius - origin[0]) / voxelSize), 0, dimensions[0] - 1);
    const minY = clamp(Math.floor((cy - radius - origin[1]) / voxelSize), 0, dimensions[1] - 1);
    const maxY = clamp(Math.floor((cy + radius - origin[1]) / voxelSize), 0, dimensions[1] - 1);
    const minZ = clamp(Math.floor((cz - radius - origin[2]) / voxelSize), 0, dimensions[2] - 1);
    const maxZ = clamp(Math.floor((cz + radius - origin[2]) / voxelSize), 0, dimensions[2] - 1);
    const qx = input.quaternions[quaternionOffset];
    const qy = input.quaternions[quaternionOffset + 1];
    const qz = input.quaternions[quaternionOffset + 2];
    const qw = input.quaternions[quaternionOffset + 3];
    const norm = Math.hypot(qx, qy, qz, qw);
    if (!(norm > 0) || !Number.isFinite(norm)) invalid(`quaternion ${index} is invalid`);
    const inverse = [-qx / norm, -qy / norm, -qz / norm, qw / norm] as const;

    for (let z = minZ; z <= maxZ; z++) {
      const dz = origin[2] + (z + 0.5) * voxelSize - cz;
      for (let y = minY; y <= maxY; y++) {
        const dy = origin[1] + (y + 0.5) * voxelSize - cy;
        for (let x = minX; x <= maxX; x++) {
          const dx = origin[0] + (x + 0.5) * voxelSize - cx;
          const [lx, ly, lz] = rotateVector(dx, dy, dz, inverse);
          if ((lx / sx) ** 2 + (ly / sy) ** 2 + (lz / sz) ** 2 <= 1) {
            occupied[voxelIndex(x, y, z, dimensions)] = 1;
          }
        }
      }
    }
  }
  return { occupied, dimensions, origin, voxelSize };
}

function retainVoxelComponents(field: VoxelField, minimumSize: number): void {
  const { occupied, dimensions } = field;
  const visited = new Uint8Array(occupied.length);
  const queue = new Int32Array(occupied.length);
  const component: number[] = [];
  for (let start = 0; start < occupied.length; start++) {
    if (occupied[start] === 0 || visited[start] !== 0) continue;
    component.length = 0;
    let read = 0;
    let write = 0;
    queue[write++] = start;
    visited[start] = 1;
    while (read < write) {
      const current = queue[read++];
      component.push(current);
      forEachVoxelNeighbor(current, dimensions, (neighbor) => {
        if (occupied[neighbor] !== 0 && visited[neighbor] === 0) {
          visited[neighbor] = 1;
          queue[write++] = neighbor;
        }
      });
    }
    if (component.length < minimumSize) {
      for (const index of component) occupied[index] = 0;
    }
  }
}

/**
 * Eliminate diagonal 2x2 occupancy patterns around grid edges. Those patterns
 * make a closed voxel boundary geometrically sealed but topologically
 * non-manifold after coincident corners are canonicalized.
 */
function regularizeVoxelTopology(field: VoxelField): void {
  const { occupied, dimensions } = field;
  const [width, height, depth] = dimensions;
  const fillCheckerboard = (indices: [number, number, number, number]): boolean => {
    const [a, b, c, d] = indices;
    const diagonalAd = occupied[a] !== 0 && occupied[d] !== 0;
    const diagonalBc = occupied[b] !== 0 && occupied[c] !== 0;
    if (
      (diagonalAd && occupied[b] === 0 && occupied[c] === 0) ||
      (diagonalBc && occupied[a] === 0 && occupied[d] === 0)
    ) {
      occupied[a] = 1;
      occupied[b] = 1;
      occupied[c] = 1;
      occupied[d] = 1;
      return true;
    }
    return false;
  };

  for (let pass = 0; pass < 8; pass++) {
    let changed = false;
    for (let x = 0; x < width; x++) {
      for (let y = 1; y < height; y++) {
        for (let z = 1; z < depth; z++) {
          changed = fillCheckerboard([
            voxelIndex(x, y - 1, z - 1, dimensions),
            voxelIndex(x, y, z - 1, dimensions),
            voxelIndex(x, y - 1, z, dimensions),
            voxelIndex(x, y, z, dimensions),
          ]) || changed;
        }
      }
    }
    for (let y = 0; y < height; y++) {
      for (let x = 1; x < width; x++) {
        for (let z = 1; z < depth; z++) {
          changed = fillCheckerboard([
            voxelIndex(x - 1, y, z - 1, dimensions),
            voxelIndex(x, y, z - 1, dimensions),
            voxelIndex(x - 1, y, z, dimensions),
            voxelIndex(x, y, z, dimensions),
          ]) || changed;
        }
      }
    }
    for (let z = 0; z < depth; z++) {
      for (let x = 1; x < width; x++) {
        for (let y = 1; y < height; y++) {
          changed = fillCheckerboard([
            voxelIndex(x - 1, y - 1, z, dimensions),
            voxelIndex(x, y - 1, z, dimensions),
            voxelIndex(x - 1, y, z, dimensions),
            voxelIndex(x, y, z, dimensions),
          ]) || changed;
        }
      }
    }
    if (!changed) break;
  }
}

function fillEnclosedVoxels(field: VoxelField): void {
  const { occupied, dimensions } = field;
  const exterior = new Uint8Array(occupied.length);
  const queue = new Int32Array(occupied.length);
  let read = 0;
  let write = 0;
  for (let z = 0; z < dimensions[2]; z++) {
    for (let y = 0; y < dimensions[1]; y++) {
      for (let x = 0; x < dimensions[0]; x++) {
        if (
          x !== 0 && x !== dimensions[0] - 1 &&
          y !== 0 && y !== dimensions[1] - 1 &&
          z !== 0 && z !== dimensions[2] - 1
        ) continue;
        const index = voxelIndex(x, y, z, dimensions);
        if (occupied[index] === 0 && exterior[index] === 0) {
          exterior[index] = 1;
          queue[write++] = index;
        }
      }
    }
  }
  while (read < write) {
    const current = queue[read++];
    forEachVoxelNeighbor(current, dimensions, (neighbor) => {
      if (occupied[neighbor] === 0 && exterior[neighbor] === 0) {
        exterior[neighbor] = 1;
        queue[write++] = neighbor;
      }
    });
  }
  for (let index = 0; index < occupied.length; index++) {
    if (occupied[index] === 0 && exterior[index] === 0) occupied[index] = 1;
  }
}

function meshVoxelBoundary(field: VoxelField): TriangleMesh {
  const { occupied, dimensions, origin, voxelSize } = field;
  const vertices: number[] = [];
  const faces: number[] = [];
  const vertexByCorner = new Map<string, number>();
  const directions = [
    [-1, 0, 0, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]],
    [1, 0, 0, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]],
    [0, -1, 0, [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
    [0, 1, 0, [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]],
    [0, 0, -1, [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]],
    [0, 0, 1, [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
  ] as const;
  const cornerIndex = (x: number, y: number, z: number): number => {
    const key = `${x},${y},${z}`;
    let index = vertexByCorner.get(key);
    if (index === undefined) {
      index = vertices.length / 3;
      vertexByCorner.set(key, index);
      vertices.push(
        origin[0] + x * voxelSize,
        origin[1] + y * voxelSize,
        origin[2] + z * voxelSize,
      );
    }
    return index;
  };

  for (let z = 0; z < dimensions[2]; z++) {
    for (let y = 0; y < dimensions[1]; y++) {
      for (let x = 0; x < dimensions[0]; x++) {
        if (occupied[voxelIndex(x, y, z, dimensions)] === 0) continue;
        for (const [dx, dy, dz, corners] of directions) {
          const nx = x + dx;
          const ny = y + dy;
          const nz = z + dz;
          if (
            nx >= 0 && nx < dimensions[0] &&
            ny >= 0 && ny < dimensions[1] &&
            nz >= 0 && nz < dimensions[2] &&
            occupied[voxelIndex(nx, ny, nz, dimensions)] !== 0
          ) continue;
          const [a, b, c, d] = corners.map(([cx, cy, cz]) =>
            cornerIndex(x + cx, y + cy, z + cz),
          );
          faces.push(a, b, c, a, c, d);
        }
      }
    }
  }
  if (faces.length === 0) invalid("Gaussian reconstruction produced no surface");
  return { vertices: new Float64Array(vertices), faces: new Int32Array(faces) };
}

function encodeTriangleMeshGlb(mesh: TriangleMesh): ArrayBuffer {
  const positions = Float32Array.from(mesh.vertices);
  const indices = Uint32Array.from(mesh.faces);
  const positionBytes = positions.byteLength;
  const indexOffset = (positionBytes + 3) & ~3;
  const binaryLength = indexOffset + indices.byteLength;
  const binary = new Uint8Array(binaryLength);
  binary.set(new Uint8Array(positions.buffer), 0);
  binary.set(new Uint8Array(indices.buffer), indexOffset);
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < positions.length; offset += 3) {
    for (let axis = 0; axis < 3; axis++) {
      mins[axis] = Math.min(mins[axis], positions[offset + axis]);
      maxs[axis] = Math.max(maxs[axis], positions[offset + axis]);
    }
  }
  const document = {
    asset: { version: "2.0", generator: "chitin gaussian-field reconstruction" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: "VEC3", min: mins, max: maxs },
      { bufferView: 1, componentType: 5125, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes, target: 34962 },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength, target: 34963 },
    ],
    buffers: [{ byteLength: binaryLength }],
  };
  const encoder = new TextEncoder();
  const json = encoder.encode(JSON.stringify(document));
  const jsonLength = (json.length + 3) & ~3;
  const totalLength = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new ArrayBuffer(totalLength);
  const view = new DataView(output);
  const bytes = new Uint8Array(output);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(json, 20);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binaryLength, true);
  view.setUint32(binaryHeader + 4, 0x004e4942, true);
  bytes.set(binary, binaryHeader + 8);
  return output;
}

function rotateVector(
  x: number,
  y: number,
  z: number,
  quaternion: readonly [number, number, number, number],
): [number, number, number] {
  const [qx, qy, qz, qw] = quaternion;
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    x + qw * tx + (qy * tz - qz * ty),
    y + qw * ty + (qz * tx - qx * tz),
    z + qw * tz + (qx * ty - qy * tx),
  ];
}

function forEachVoxelNeighbor(
  index: number,
  dimensions: [number, number, number],
  callback: (neighbor: number) => void,
): void {
  const plane = dimensions[0] * dimensions[1];
  const z = Math.floor(index / plane);
  const remainder = index - z * plane;
  const y = Math.floor(remainder / dimensions[0]);
  const x = remainder - y * dimensions[0];
  if (x > 0) callback(index - 1);
  if (x + 1 < dimensions[0]) callback(index + 1);
  if (y > 0) callback(index - dimensions[0]);
  if (y + 1 < dimensions[1]) callback(index + dimensions[0]);
  if (z > 0) callback(index - plane);
  if (z + 1 < dimensions[2]) callback(index + plane);
}

function voxelIndex(
  x: number,
  y: number,
  z: number,
  dimensions: [number, number, number],
): number {
  return x + dimensions[0] * (y + dimensions[1] * z);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function invalid(message: string): never {
  throw new ChitinError("INVALID_MESH", `Gaussian field: ${message}`, {
    stage: "parsing-input",
    suggestion: "Check Gaussian centers, scales, rotations, and reconstruction options.",
  });
}
