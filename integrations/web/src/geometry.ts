import type { PhysFile, PhysHull } from "./phys-parser.js";
import { selectLodHulls } from "./phys-parser.js";

export type Vec3 = readonly [number, number, number];
export type Quaternion = readonly [number, number, number, number];

export interface PhysTransform {
  translation?: Vec3;
  rotation?: Quaternion;
  scale?: Vec3;
}

export interface ResolvedPhysTransform {
  translation: Vec3;
  rotation: Quaternion;
  scale: Vec3;
}

export interface PhysBounds {
  min: [number, number, number];
  max: [number, number, number];
}

export interface MaterializedPhysHull {
  vertices: Float32Array;
  indices: Uint16Array;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  boneIndex: number | null;
  sourceHullIndex: number;
}

export interface MaterializedPhysGeometry {
  hulls: MaterializedPhysHull[];
  bounds: PhysBounds | null;
}

export interface MaterializeOptions {
  /** Choose the LOD tier nearest this concavity. Omit for LOD 0. */
  lodConcavity?: number;
  /**
   * `bind` bakes bone-local hulls into asset-local bind-pose coordinates.
   * `stored` preserves the coordinates exactly as stored in `.phys`.
   * Defaults to `bind` because static runtime consumers need one asset space.
   */
  pose?: "stored" | "bind";
}

export interface CombinedHullRange {
  sourceHullIndex: number;
  boneIndex: number | null;
  indexStart: number;
  indexCount: number;
}

export interface CombinedPhysGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
  ranges: CombinedHullRange[];
  bounds: PhysBounds | null;
}

/**
 * Select and materialize hulls without depending on a renderer or physics API.
 * The returned indices retain their per-hull locality; use combinePhysGeometry
 * when a consumer needs one shared vertex/index buffer.
 */
export function materializePhysHulls(
  phys: PhysFile,
  options?: MaterializeOptions,
): MaterializedPhysGeometry {
  const hulls =
    options?.lodConcavity === undefined
      ? phys.hulls
      : selectLodHulls(phys, options.lodConcavity);
  const pose = options?.pose ?? "bind";

  const materialized = hulls.map((hull, sourceHullIndex) => {
    const vertices = materializeHullVertices(phys, hull, sourceHullIndex, pose);
    const bounds = calculateVertexBounds(vertices);
    if (bounds === null) {
      throw new Error(`hull ${sourceHullIndex}: contains no vertices`);
    }
    return {
      vertices,
      indices: hull.indices,
      aabbMin: bounds.min,
      aabbMax: bounds.max,
      boneIndex: hull.boneIndex,
      sourceHullIndex,
    };
  });

  return {
    hulls: materialized,
    bounds: combineBounds(materialized),
  };
}

function materializeHullVertices(
  phys: PhysFile,
  hull: PhysHull,
  hullIndex: number,
  pose: "stored" | "bind",
): Float32Array {
  if (pose === "stored" || hull.boneIndex === null) {
    return hull.vertices;
  }
  if (!phys.hasBindPoses) {
    throw new Error(
      `hull ${hullIndex} is bone-local (bone ${hull.boneIndex}) but the ` +
        `.phys carries no bind poses`,
    );
  }
  const bone = phys.bones[hull.boneIndex];
  if (bone === undefined) {
    throw new Error(
      `hull ${hullIndex}: bone_index ${hull.boneIndex} has no bind transform`,
    );
  }
  return applyBindPose(hull.vertices, bone.bindTransform);
}

/** Apply the `.phys` row-vector bind transform to packed xyz vertices. */
export function applyBindPose(
  vertices: Float32Array,
  bindTransform: Float32Array,
): Float32Array {
  if (bindTransform.length !== 16 || !bindTransform.every(Number.isFinite)) {
    throw new Error("bind transform must contain 16 finite values");
  }
  const m = bindTransform;
  const out = new Float32Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i];
    const y = vertices[i + 1];
    const z = vertices[i + 2];
    out[i] = x * m[0] + y * m[4] + z * m[8] + m[12];
    out[i + 1] = x * m[1] + y * m[5] + z * m[9] + m[13];
    out[i + 2] = x * m[2] + y * m[6] + z * m[10] + m[14];
  }
  return out;
}

/** Apply scale, then quaternion rotation, then translation to packed xyz. */
export function transformHullVertices(
  vertices: Float32Array,
  transform?: PhysTransform,
): Float32Array {
  if (transform === undefined) return vertices;
  const { translation, scale, rotation } = resolvePhysTransform(transform);
  const [qx, qy, qz, qw] = rotation;
  const out = new Float32Array(vertices.length);

  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i] * scale[0];
    const y = vertices[i + 1] * scale[1];
    const z = vertices[i + 2] * scale[2];

    // Quaternion-vector rotation, expanded to avoid runtime-specific math types.
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    out[i] = x + qw * tx + (qy * tz - qz * ty) + translation[0];
    out[i + 1] = y + qw * ty + (qz * tx - qx * tz) + translation[1];
    out[i + 2] = z + qw * tz + (qx * ty - qy * tx) + translation[2];
  }
  return out;
}

/** Validate defaults and normalize rotation once for runtime adapters. */
export function resolvePhysTransform(
  transform?: PhysTransform,
): ResolvedPhysTransform {
  const translation = validateVec3(
    transform?.translation ?? [0, 0, 0],
    "translation",
  );
  const scale = validateVec3(transform?.scale ?? [1, 1, 1], "scale");
  if (scale.some((component) => component === 0)) {
    throw new Error("scale components must be non-zero");
  }
  const rotation = normalizeQuaternion(transform?.rotation ?? [0, 0, 0, 1]);
  return { translation, rotation, scale };
}

/** Combine per-hull buffers while retaining ranges back to their source hulls. */
export function combinePhysGeometry(
  geometry: MaterializedPhysGeometry,
): CombinedPhysGeometry {
  const vertexCount = geometry.hulls.reduce(
    (total, hull) => total + hull.vertices.length / 3,
    0,
  );
  const indexCount = geometry.hulls.reduce(
    (total, hull) => total + hull.indices.length,
    0,
  );
  const vertices = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(indexCount);
  const ranges: CombinedHullRange[] = [];
  let vertexOffset = 0;
  let indexOffset = 0;

  for (const hull of geometry.hulls) {
    vertices.set(hull.vertices, vertexOffset * 3);
    for (let i = 0; i < hull.indices.length; i++) {
      indices[indexOffset + i] = hull.indices[i] + vertexOffset;
    }
    ranges.push({
      sourceHullIndex: hull.sourceHullIndex,
      boneIndex: hull.boneIndex,
      indexStart: indexOffset,
      indexCount: hull.indices.length,
    });
    vertexOffset += hull.vertices.length / 3;
    indexOffset += hull.indices.length;
  }

  return { vertices, indices, ranges, bounds: geometry.bounds };
}

export function calculateVertexBounds(
  vertices: Float32Array,
): PhysBounds | null {
  if (vertices.length === 0) return null;
  if (vertices.length % 3 !== 0) {
    throw new Error("vertex array length must be a multiple of 3");
  }
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const value = vertices[i + axis];
      if (!Number.isFinite(value)) {
        throw new Error(`vertex ${i / 3} contains a non-finite coordinate`);
      }
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }
  return { min, max };
}

function combineBounds(hulls: MaterializedPhysHull[]): PhysBounds | null {
  if (hulls.length === 0) return null;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const hull of hulls) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], hull.aabbMin[axis]);
      max[axis] = Math.max(max[axis], hull.aabbMax[axis]);
    }
  }
  return { min, max };
}

function validateVec3(value: Vec3, label: string): Vec3 {
  if (value.length !== 3 || !value.every(Number.isFinite)) {
    throw new Error(`${label} must contain 3 finite values`);
  }
  return value;
}

function normalizeQuaternion(value: Quaternion): Quaternion {
  if (value.length !== 4 || !value.every(Number.isFinite)) {
    throw new Error("rotation must contain 4 finite values");
  }
  const length = Math.hypot(value[0], value[1], value[2], value[3]);
  if (length === 0) throw new Error("rotation quaternion must be non-zero");
  return [
    value[0] / length,
    value[1] / length,
    value[2] / length,
    value[3] / length,
  ];
}
