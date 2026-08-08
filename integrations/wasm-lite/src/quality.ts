import { splitMeshComponents, type TriangleMesh } from "./mesh.js";
import type { ConvexHull } from "./types.js";
import {
  meshBounds,
  boundsDiagonal,
  boundsVolume,
  containsBounds,
  triangleArea,
  componentArea,
  type Bounds,
  type Point,
} from "./geometry.js";

export interface ColliderQualityOptions {
  /** Deterministic, area-weighted samples taken from the source surface. Default 2048. */
  surfaceSamples?: number;
  /** Deterministic samples taken from the collider AABB. Default 4096. */
  volumeSamples?: number;
  /** Absolute tolerance used for volume containment. Default: source diagonal * 1e-5. */
  tolerance?: number;
  /** Surface-fit tolerance relative to each connected component's diagonal. Default 0.02. */
  surfaceToleranceRatio?: number;
  /** Minimum collider-occupied samples required for volume metrics. Default 32. */
  minColliderSamples?: number;
  /** Free-space penetration deeper than this component-diagonal ratio is severe. Default 0.02. */
  deepFillClearanceRatio?: number;
}

export interface ColliderQualityResult {
  method: "deterministic_halton_v1";
  source_surface_coverage: number;
  worst_component_surface_coverage: number;
  collider_volume_precision: number | null;
  false_fill_fraction: number | null;
  deep_false_fill_fraction: number | null;
  surface_samples: number;
  volume_samples: number;
  collider_volume_samples: number;
  component_count: number;
  volume_tolerance: number;
  surface_tolerance_ratio: number;
  deep_fill_clearance_ratio: number;
  components: ColliderQualityComponentResult[];
}

export interface ColliderQualityComponentResult {
  component_index: number;
  vertex_count: number;
  triangle_count: number;
  surface_area_fraction: number;
  diagonal_ratio: number;
  surface_samples: number;
  surface_coverage: number;
  hull_count: number | null;
  collider_triangle_count: number | null;
  collider_volume_precision: number | null;
  false_fill_fraction: number | null;
  deep_false_fill_fraction: number | null;
  collider_volume_samples: number | null;
}

interface Plane {
  x: number;
  y: number;
  z: number;
  d: number;
}

interface PreparedHull {
  bounds: Bounds;
  planes: Plane[];
}

interface PreparedComponent {
  mesh: TriangleMesh;
  bounds: Bounds;
  area: number;
}

const RAY_DIRECTION: Point = (() => {
  const direction: Point = [1, 0.372013, 0.529117];
  const length = Math.hypot(...direction);
  return [direction[0] / length, direction[1] / length, direction[2] / length];
})();

function positiveInteger(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < 1) {
    throw new Error(`${label} must be a positive integer, got ${resolved}`);
  }
  return resolved;
}

function prepareHull(hull: ConvexHull): PreparedHull {
  const bounds = meshBounds(hull.vertices);
  const center: Point = [0, 0, 0];
  const vertexCount = hull.vertices.length / 3;
  for (let offset = 0; offset < hull.vertices.length; offset += 3) {
    center[0] += hull.vertices[offset];
    center[1] += hull.vertices[offset + 1];
    center[2] += hull.vertices[offset + 2];
  }
  if (vertexCount > 0) {
    center[0] /= vertexCount;
    center[1] /= vertexCount;
    center[2] /= vertexCount;
  }
  const planes: Plane[] = [];
  for (let offset = 0; offset < hull.indices.length; offset += 3) {
    const ai = hull.indices[offset] * 3;
    const bi = hull.indices[offset + 1] * 3;
    const ci = hull.indices[offset + 2] * 3;
    const ax = hull.vertices[ai];
    const ay = hull.vertices[ai + 1];
    const az = hull.vertices[ai + 2];
    const abx = hull.vertices[bi] - ax;
    const aby = hull.vertices[bi + 1] - ay;
    const abz = hull.vertices[bi + 2] - az;
    const acx = hull.vertices[ci] - ax;
    const acy = hull.vertices[ci + 1] - ay;
    const acz = hull.vertices[ci + 2] - az;
    let x = aby * acz - abz * acy;
    let y = abz * acx - abx * acz;
    let z = abx * acy - aby * acx;
    const length = Math.hypot(x, y, z);
    if (length === 0) continue;
    x /= length;
    y /= length;
    z /= length;
    let d = -(x * ax + y * ay + z * az);
    if (x * center[0] + y * center[1] + z * center[2] + d > 0) {
      x = -x;
      y = -y;
      z = -z;
      d = -d;
    }
    planes.push({ x, y, z, d });
  }
  return { bounds, planes };
}

function insideHull(hull: PreparedHull, point: Point, tolerance: number): boolean {
  if (hull.planes.length === 0 || !containsBounds(hull.bounds, point, tolerance)) return false;
  return hull.planes.every((plane) =>
    plane.x * point[0] + plane.y * point[1] + plane.z * point[2] + plane.d <= tolerance
  );
}

function insideAnyHull(hulls: PreparedHull[], point: Point, tolerance: number): boolean {
  return hulls.some((hull) => insideHull(hull, point, tolerance));
}

function radicalInverse(index: number, base: number): number {
  let result = 0;
  let fraction = 1 / base;
  while (index > 0) {
    result += (index % base) * fraction;
    index = Math.floor(index / base);
    fraction /= base;
  }
  return result;
}

function rayTriangleDistance(point: Point, mesh: TriangleMesh, faceOffset: number): number | null {
  const ai = mesh.faces[faceOffset] * 3;
  const bi = mesh.faces[faceOffset + 1] * 3;
  const ci = mesh.faces[faceOffset + 2] * 3;
  const ax = mesh.vertices[ai];
  const ay = mesh.vertices[ai + 1];
  const az = mesh.vertices[ai + 2];
  const edge1x = mesh.vertices[bi] - ax;
  const edge1y = mesh.vertices[bi + 1] - ay;
  const edge1z = mesh.vertices[bi + 2] - az;
  const edge2x = mesh.vertices[ci] - ax;
  const edge2y = mesh.vertices[ci + 1] - ay;
  const edge2z = mesh.vertices[ci + 2] - az;
  const px = RAY_DIRECTION[1] * edge2z - RAY_DIRECTION[2] * edge2y;
  const py = RAY_DIRECTION[2] * edge2x - RAY_DIRECTION[0] * edge2z;
  const pz = RAY_DIRECTION[0] * edge2y - RAY_DIRECTION[1] * edge2x;
  const determinant = edge1x * px + edge1y * py + edge1z * pz;
  if (Math.abs(determinant) < 1e-12) return null;
  const inverse = 1 / determinant;
  const tx = point[0] - ax;
  const ty = point[1] - ay;
  const tz = point[2] - az;
  const u = (tx * px + ty * py + tz * pz) * inverse;
  if (u < 0 || u > 1) return null;
  const qx = ty * edge1z - tz * edge1y;
  const qy = tz * edge1x - tx * edge1z;
  const qz = tx * edge1y - ty * edge1x;
  const v = (RAY_DIRECTION[0] * qx + RAY_DIRECTION[1] * qy + RAY_DIRECTION[2] * qz) * inverse;
  if (v < 0 || u + v > 1) return null;
  const distance = (edge2x * qx + edge2y * qy + edge2z * qz) * inverse;
  return distance > 0 ? distance : null;
}

function pointTriangleDistanceSquared(point: Point, mesh: TriangleMesh, faceOffset: number): number {
  const ai = mesh.faces[faceOffset] * 3;
  const bi = mesh.faces[faceOffset + 1] * 3;
  const ci = mesh.faces[faceOffset + 2] * 3;
  const ax = mesh.vertices[ai];
  const ay = mesh.vertices[ai + 1];
  const az = mesh.vertices[ai + 2];
  const bx = mesh.vertices[bi];
  const by = mesh.vertices[bi + 1];
  const bz = mesh.vertices[bi + 2];
  const cx = mesh.vertices[ci];
  const cy = mesh.vertices[ci + 1];
  const cz = mesh.vertices[ci + 2];
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const apx = point[0] - ax;
  const apy = point[1] - ay;
  const apz = point[2] - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;

  const bpx = point[0] - bx;
  const bpy = point[1] - by;
  const bpz = point[2] - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const dx = point[0] - (ax + v * abx);
    const dy = point[1] - (ay + v * aby);
    const dz = point[2] - (az + v * abz);
    return dx * dx + dy * dy + dz * dz;
  }

  const cpx = point[0] - cx;
  const cpy = point[1] - cy;
  const cpz = point[2] - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const dx = point[0] - (ax + w * acx);
    const dy = point[1] - (ay + w * acy);
    const dz = point[2] - (az + w * acz);
    return dx * dx + dy * dy + dz * dz;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const edgeX = cx - bx;
    const edgeY = cy - by;
    const edgeZ = cz - bz;
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    const dx = point[0] - (bx + w * edgeX);
    const dy = point[1] - (by + w * edgeY);
    const dz = point[2] - (bz + w * edgeZ);
    return dx * dx + dy * dy + dz * dz;
  }

  const denominator = 1 / (va + vb + vc);
  const v = vb * denominator;
  const w = vc * denominator;
  const dx = point[0] - (ax + abx * v + acx * w);
  const dy = point[1] - (ay + aby * v + acy * w);
  const dz = point[2] - (az + abz * v + acz * w);
  return dx * dx + dy * dy + dz * dz;
}

function pointMeshDistanceSquared(point: Point, mesh: TriangleMesh): number {
  let minimum = Infinity;
  for (let offset = 0; offset < mesh.faces.length; offset += 3) {
    minimum = Math.min(minimum, pointTriangleDistanceSquared(point, mesh, offset));
  }
  return minimum;
}

function insideComponent(component: PreparedComponent, point: Point, tolerance: number): boolean {
  if (!containsBounds(component.bounds, point, tolerance)) return false;
  const hits: number[] = [];
  for (let offset = 0; offset < component.mesh.faces.length; offset += 3) {
    const distance = rayTriangleDistance(point, component.mesh, offset);
    if (distance !== null) hits.push(distance);
  }
  if (hits.length === 0) return false;
  hits.sort((left, right) => left - right);
  let uniqueHits = 0;
  let previous = -Infinity;
  for (const hit of hits) {
    if (hit - previous > tolerance) {
      uniqueHits++;
      previous = hit;
    }
  }
  return uniqueHits % 2 === 1;
}

function insideSource(components: PreparedComponent[], point: Point, tolerance: number): boolean {
  return components.some((component) => insideComponent(component, point, tolerance));
}

function allocatedWeightedSamples(weights: number[], requested: number): number[] {
  const total = Math.max(requested, weights.length);
  const allocation = weights.map(() => 1);
  const distributable = total - weights.length;
  let assigned = 0;
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const fractions: Array<{ index: number; fraction: number }> = [];
  for (let index = 0; index < weights.length; index++) {
    const exact = totalWeight > 0
      ? distributable * weights[index] / totalWeight
      : distributable / weights.length;
    const whole = Math.floor(exact);
    allocation[index] += whole;
    assigned += whole;
    fractions.push({ index, fraction: exact - whole });
  }
  fractions.sort((left, right) => right.fraction - left.fraction || left.index - right.index);
  for (let index = 0; index < distributable - assigned; index++) allocation[fractions[index].index]++;
  return allocation;
}

function allocatedSamples(components: PreparedComponent[], requested: number): number[] {
  return allocatedWeightedSamples(components.map((component) => component.area), requested);
}

function sampleComponentSurface(
  component: PreparedComponent,
  count: number,
  hulls: PreparedHull[],
  tolerance: number,
): number {
  const cumulative: number[] = [];
  let totalArea = 0;
  for (let offset = 0; offset < component.mesh.faces.length; offset += 3) {
    totalArea += triangleArea(
      component.mesh.vertices,
      component.mesh.faces[offset],
      component.mesh.faces[offset + 1],
      component.mesh.faces[offset + 2],
    );
    cumulative.push(totalArea);
  }
  if (totalArea === 0) return 0;
  let covered = 0;
  let triangle = 0;
  for (let sample = 0; sample < count; sample++) {
    const target = (sample + 0.5) * totalArea / count;
    while (triangle < cumulative.length - 1 && cumulative[triangle] < target) triangle++;
    const offset = triangle * 3;
    const ai = component.mesh.faces[offset] * 3;
    const bi = component.mesh.faces[offset + 1] * 3;
    const ci = component.mesh.faces[offset + 2] * 3;
    const u = radicalInverse(sample + 1, 2);
    const v = radicalInverse(sample + 1, 3);
    const root = Math.sqrt(u);
    const wa = 1 - root;
    const wb = root * (1 - v);
    const wc = root * v;
    const point: Point = [
      wa * component.mesh.vertices[ai] + wb * component.mesh.vertices[bi] + wc * component.mesh.vertices[ci],
      wa * component.mesh.vertices[ai + 1] + wb * component.mesh.vertices[bi + 1] + wc * component.mesh.vertices[ci + 1],
      wa * component.mesh.vertices[ai + 2] + wb * component.mesh.vertices[bi + 2] + wc * component.mesh.vertices[ci + 2],
    ];
    if (insideAnyHull(hulls, point, tolerance)) covered++;
  }
  return covered;
}

function rounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function combinedHullBounds(hulls: PreparedHull[], fallback: Bounds): Bounds {
  if (hulls.length === 0) return fallback;
  return hulls.reduce<Bounds>((combined, hull) => ({
    min: [
      Math.min(combined.min[0], hull.bounds.min[0]),
      Math.min(combined.min[1], hull.bounds.min[1]),
      Math.min(combined.min[2], hull.bounds.min[2]),
    ],
    max: [
      Math.max(combined.max[0], hull.bounds.max[0]),
      Math.max(combined.max[1], hull.bounds.max[1]),
      Math.max(combined.max[2], hull.bounds.max[2]),
    ],
  }), hulls[0].bounds);
}

/**
 * Deterministically sample artifact-level geometric fit.
 *
 * This is an acceptance/benchmark signal, not an exact volume proof. Surface
 * samples measure whether source geometry is represented by a collider. Volume
 * samples estimate how much collider-occupied space is real source volume,
 * which makes bridged cavities and oversized hulls visible as false fill.
 */
export function evaluateColliderQuality(
  source: TriangleMesh,
  hulls: ConvexHull[],
  options: ColliderQualityOptions = {},
  componentHulls?: ConvexHull[][],
): ColliderQualityResult {
  const requestedSurfaceSamples = positiveInteger(options.surfaceSamples, 2048, "surfaceSamples");
  const volumeSamples = positiveInteger(options.volumeSamples, 4096, "volumeSamples");
  const minColliderSamples = positiveInteger(options.minColliderSamples, 32, "minColliderSamples");
  const deepFillClearanceRatio = options.deepFillClearanceRatio ?? 0.02;
  if (
    !Number.isFinite(deepFillClearanceRatio) ||
    deepFillClearanceRatio < 0 ||
    deepFillClearanceRatio > 1
  ) {
    throw new Error(
      `deepFillClearanceRatio must be a finite number in [0, 1], got ${deepFillClearanceRatio}`,
    );
  }
  const sourceBounds = meshBounds(source.vertices);
  const tolerance = options.tolerance ?? Math.max(boundsDiagonal(sourceBounds) * 1e-5, 1e-9);
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error(`tolerance must be a finite non-negative number, got ${tolerance}`);
  }
  const surfaceToleranceRatio = options.surfaceToleranceRatio ?? 0.02;
  if (!Number.isFinite(surfaceToleranceRatio) || surfaceToleranceRatio < 0 || surfaceToleranceRatio > 1) {
    throw new Error(
      `surfaceToleranceRatio must be a finite number in [0, 1], got ${surfaceToleranceRatio}`,
    );
  }
  const components = splitMeshComponents(source.vertices, source.faces).map((mesh) => ({
    mesh,
    bounds: meshBounds(mesh.vertices),
    area: componentArea(mesh),
  }));
  const preparedHulls = hulls.map(prepareHull).filter((hull) => hull.planes.length > 0);
  const preparedComponentHulls = componentHulls?.length === components.length
    ? componentHulls.map((items) => items.map(prepareHull).filter((hull) => hull.planes.length > 0))
    : null;
  const allocation = allocatedSamples(components, requestedSurfaceSamples);
  let coveredSurfaceSamples = 0;
  let actualSurfaceSamples = 0;
  let worstComponentCoverage = 1;
  const totalSurfaceArea = components.reduce((sum, component) => sum + component.area, 0);
  const sourceDiagonal = boundsDiagonal(sourceBounds);
  const componentResults: ColliderQualityComponentResult[] = [];
  for (let index = 0; index < components.length; index++) {
    const count = allocation[index];
    const surfaceTolerance = Math.max(
      tolerance,
      boundsDiagonal(components[index].bounds) * surfaceToleranceRatio,
    );
    const covered = sampleComponentSurface(
      components[index],
      count,
      preparedComponentHulls?.[index] ?? preparedHulls,
      surfaceTolerance,
    );
    coveredSurfaceSamples += covered;
    actualSurfaceSamples += count;
    const coverage = covered / count;
    worstComponentCoverage = Math.min(worstComponentCoverage, coverage);
    componentResults.push({
      component_index: index,
      vertex_count: components[index].mesh.vertices.length / 3,
      triangle_count: components[index].mesh.faces.length / 3,
      surface_area_fraction: rounded(
        totalSurfaceArea > 0 ? components[index].area / totalSurfaceArea : 0,
      ),
      diagonal_ratio: rounded(
        sourceDiagonal > 0 ? boundsDiagonal(components[index].bounds) / sourceDiagonal : 1,
      ),
      surface_samples: count,
      surface_coverage: rounded(coverage),
      hull_count: componentHulls?.[index]?.length ?? null,
      collider_triangle_count: componentHulls?.[index]?.reduce(
        (sum, hull) => sum + hull.indices.length / 3,
        0,
      ) ?? null,
      collider_volume_precision: null,
      false_fill_fraction: null,
      deep_false_fill_fraction: null,
      collider_volume_samples: null,
    });
  }

  let colliderVolumeSamples = 0;
  let trueVolumeSamples = 0;
  let deepFalseFillSamples = 0;
  if (preparedComponentHulls) {
    const componentColliderBounds = preparedComponentHulls.map((ownedHulls, index) =>
      combinedHullBounds(ownedHulls, components[index].bounds)
    );
    const volumeAllocation = allocatedWeightedSamples(
      componentColliderBounds.map(boundsVolume),
      volumeSamples,
    );
    for (let componentIndex = 0; componentIndex < components.length; componentIndex++) {
      const ownedHulls = preparedComponentHulls[componentIndex];
      const sampleCount = volumeAllocation[componentIndex];
      const colliderBounds = componentColliderBounds[componentIndex];
      let componentColliderSamples = 0;
      let componentTrueSamples = 0;
      let componentDeepFalseFillSamples = 0;
      const deepClearanceSquared = (
        boundsDiagonal(components[componentIndex].bounds) * deepFillClearanceRatio
      ) ** 2;
      for (let sample = 1; sample <= sampleCount; sample++) {
        const sequenceIndex = sample + componentIndex * sampleCount;
        const point: Point = [
          colliderBounds.min[0] + radicalInverse(sequenceIndex, 2) * (colliderBounds.max[0] - colliderBounds.min[0]),
          colliderBounds.min[1] + radicalInverse(sequenceIndex, 3) * (colliderBounds.max[1] - colliderBounds.min[1]),
          colliderBounds.min[2] + radicalInverse(sequenceIndex, 5) * (colliderBounds.max[2] - colliderBounds.min[2]),
        ];
        if (!insideAnyHull(ownedHulls, point, tolerance)) continue;
        componentColliderSamples++;
        if (insideComponent(components[componentIndex], point, tolerance)) componentTrueSamples++;
        else if (
          pointMeshDistanceSquared(point, components[componentIndex].mesh) > deepClearanceSquared
        ) {
          componentDeepFalseFillSamples++;
        }
      }
      colliderVolumeSamples += componentColliderSamples;
      trueVolumeSamples += componentTrueSamples;
      deepFalseFillSamples += componentDeepFalseFillSamples;
      const componentPrecision = componentColliderSamples >= Math.min(minColliderSamples, 8)
        ? componentTrueSamples / componentColliderSamples
        : null;
      componentResults[componentIndex].collider_volume_precision = componentPrecision === null
        ? null
        : rounded(componentPrecision);
      componentResults[componentIndex].false_fill_fraction = componentPrecision === null
        ? null
        : rounded(1 - componentPrecision);
      componentResults[componentIndex].deep_false_fill_fraction = componentPrecision === null
        ? null
        : rounded(componentDeepFalseFillSamples / componentColliderSamples);
      componentResults[componentIndex].collider_volume_samples = componentColliderSamples;
    }
  } else {
    const colliderBounds = combinedHullBounds(preparedHulls, sourceBounds);
    for (let sample = 1; sample <= volumeSamples; sample++) {
      const point: Point = [
        colliderBounds.min[0] + radicalInverse(sample, 2) * (colliderBounds.max[0] - colliderBounds.min[0]),
        colliderBounds.min[1] + radicalInverse(sample, 3) * (colliderBounds.max[1] - colliderBounds.min[1]),
        colliderBounds.min[2] + radicalInverse(sample, 5) * (colliderBounds.max[2] - colliderBounds.min[2]),
      ];
      if (!insideAnyHull(preparedHulls, point, tolerance)) continue;
      colliderVolumeSamples++;
      if (insideSource(components, point, tolerance)) trueVolumeSamples++;
      else {
        const clearanceSquared = (boundsDiagonal(sourceBounds) * deepFillClearanceRatio) ** 2;
        const distanceSquared = components.reduce(
          (minimum, component) => Math.min(
            minimum,
            pointMeshDistanceSquared(point, component.mesh),
          ),
          Infinity,
        );
        if (distanceSquared > clearanceSquared) deepFalseFillSamples++;
      }
    }
  }
  const hasVolumeSignal = colliderVolumeSamples >= minColliderSamples;
  const volumePrecision = hasVolumeSignal ? trueVolumeSamples / colliderVolumeSamples : null;
  return {
    method: "deterministic_halton_v1",
    source_surface_coverage: rounded(actualSurfaceSamples > 0 ? coveredSurfaceSamples / actualSurfaceSamples : 0),
    worst_component_surface_coverage: rounded(worstComponentCoverage),
    collider_volume_precision: volumePrecision === null ? null : rounded(volumePrecision),
    false_fill_fraction: volumePrecision === null ? null : rounded(1 - volumePrecision),
    deep_false_fill_fraction: volumePrecision === null
      ? null
      : rounded(deepFalseFillSamples / colliderVolumeSamples),
    surface_samples: actualSurfaceSamples,
    volume_samples: volumeSamples,
    collider_volume_samples: colliderVolumeSamples,
    component_count: components.length,
    volume_tolerance: tolerance,
    surface_tolerance_ratio: surfaceToleranceRatio,
    deep_fill_clearance_ratio: deepFillClearanceRatio,
    components: componentResults,
  };
}
