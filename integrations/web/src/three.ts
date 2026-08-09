import {
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  Quaternion as ThreeQuaternion,
  Uint16BufferAttribute,
  Uint32BufferAttribute,
  Vector3,
} from "three";
import type { PhysFile } from "./phys-parser.js";
import {
  combinePhysGeometry,
  materializePhysHulls,
  type MaterializeOptions,
  type PhysTransform,
} from "./geometry.js";

export { createDeferredHullLightingPass } from "./deferred.js";
export type {
  DeferredHullLight,
  DeferredHullLightingOptions,
  DeferredHullLightingPass,
  DeferredHullLightOptions,
} from "./deferred.js";

export interface DebugOptions {
  color?: number;
  opacity?: number;
  wireframe?: boolean;
  lodConcavity?: number;
}

export interface HullGeometryOptions extends MaterializeOptions {}

export interface ShadowProxyOptions {
  lodConcavity?: number;
  mode?: "collision-lod" | "merged-hulls";
}

/** Build production Three.js geometry from the engine-neutral hull layer. */
export function createHullGeometries(
  phys: PhysFile,
  options?: HullGeometryOptions,
): BufferGeometry[] {
  return materializePhysHulls(phys, options).hulls.map((hull) => {
    const geometry = new BufferGeometry();
    geometry.setAttribute(
      "position",
      new Float32BufferAttribute(hull.vertices, 3),
    );
    geometry.setIndex(new Uint16BufferAttribute(hull.indices, 1));
    geometry.computeVertexNormals();
    return geometry;
  });
}

/**
 * Create color-invisible Three.js meshes that remain visible to shadow passes.
 * Whether a splat can receive the resulting shadow is renderer-specific.
 */
export function createShadowProxy(
  phys: PhysFile,
  options?: ShadowProxyOptions,
): Object3D {
  const group = new Group();
  group.name = "chitin_shadow_proxy";

  if (options?.mode === "merged-hulls") {
    const prepared = materializePhysHulls(phys, {
      lodConcavity: options.lodConcavity,
      pose: "bind",
    });
    if (prepared.hulls.length === 0) return group;
    const combined = combinePhysGeometry(prepared);
    const material = shadowMaterial();
    const geometry = new BufferGeometry();
    geometry.setAttribute(
      "position",
      new Float32BufferAttribute(combined.vertices, 3),
    );
    geometry.setIndex(new Uint32BufferAttribute(combined.indices, 1));
    geometry.computeVertexNormals();
    const mesh = new Mesh(geometry, material);
    mesh.name = "chitin_shadow_proxy_merged";
    mesh.castShadow = true;
    mesh.receiveShadow = false;
    group.add(mesh);
    return group;
  }

  const geometries = createHullGeometries(phys, {
    lodConcavity: options?.lodConcavity,
    pose: "bind",
  });
  if (geometries.length === 0) return group;
  const material = shadowMaterial();
  geometries.forEach((geometry, index) => {
    const mesh = new Mesh(geometry, material);
    mesh.name = `chitin_shadow_hull_${index}`;
    mesh.castShadow = true;
    mesh.receiveShadow = false;
    group.add(mesh);
  });
  return group;
}

/** Convert an Object3D's world TRS to the engine-neutral Chitin transform. */
export function physTransformFromObject3D(object: Object3D): PhysTransform {
  object.updateWorldMatrix(true, false);
  const translation = new Vector3();
  const rotation = new ThreeQuaternion();
  const scale = new Vector3();
  object.matrixWorld.decompose(translation, rotation, scale);
  return {
    translation: [translation.x, translation.y, translation.z],
    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
    scale: [scale.x, scale.y, scale.z],
  };
}

function shadowMaterial(): MeshBasicMaterial {
  return new MeshBasicMaterial({
    colorWrite: false,
    depthWrite: false,
  });
}

export function createDebugMeshes(
  phys: PhysFile,
  options?: DebugOptions,
): Group {
  const color = options?.color ?? 0x00ff88;
  const opacity = options?.opacity ?? 0.3;
  const wireframe = options?.wireframe ?? true;

  const group = new Group();
  group.name = "chitin_colliders";
  const hulls = materializePhysHulls(phys, {
    lodConcavity: options?.lodConcavity,
    pose: "bind",
  }).hulls;

  for (let i = 0; i < hulls.length; i++) {
    const hull = hulls[i];
    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(hull.vertices, 3));
    geo.setIndex(new Uint16BufferAttribute(hull.indices, 1));
    geo.computeVertexNormals();

    if (wireframe) {
      const edges = extractEdges(hull.indices);
      const edgeGeo = new BufferGeometry();
      edgeGeo.setAttribute(
        "position",
        new Float32BufferAttribute(hull.vertices, 3),
      );
      edgeGeo.setIndex(edges);
      const line = new LineSegments(edgeGeo, new LineBasicMaterial({ color }));
      line.name = `hull_${i}_wire`;
      group.add(line);
    }

    const mat = new MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    });
    const mesh = new Mesh(geo, mat);
    mesh.name = `hull_${i}`;
    group.add(mesh);
  }

  return group;
}

function extractEdges(indices: Uint16Array): number[] {
  const edges: number[] = [];
  for (let i = 0; i < indices.length; i += 3) {
    edges.push(indices[i], indices[i + 1]);
    edges.push(indices[i + 1], indices[i + 2]);
    edges.push(indices[i + 2], indices[i]);
  }
  return edges;
}
