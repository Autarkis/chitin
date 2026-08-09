import type RAPIER from "@dimforge/rapier3d-compat";
import type { PhysFile, PhysHull } from "./phys-parser.js";
import {
  applyBindPose,
  materializePhysHulls,
  resolvePhysTransform,
  transformHullVertices,
  type PhysTransform,
} from "./geometry.js";

export { applyBindPose } from "./geometry.js";

export interface ColliderResult {
  colliders: RAPIER.ColliderDesc[];
  boneMap: Map<number, RAPIER.ColliderDesc[]>;
}

export interface ColliderOptions {
  // Choose the LOD tier nearest this concavity. Omit for LOD 0 (highest detail).
  lodConcavity?: number;
}

export interface AddToWorldOptions extends ColliderOptions {
  /** Full asset transform. Scale is baked into vertices; TR go on the body. */
  transform?: PhysTransform;
}

export function createColliders(
  rapier: typeof RAPIER,
  phys: PhysFile,
  opts?: ColliderOptions,
): ColliderResult {
  const colliders: RAPIER.ColliderDesc[] = [];
  const boneMap = new Map<number, RAPIER.ColliderDesc[]>();

  const hulls = materializePhysHulls(phys, {
    lodConcavity: opts?.lodConcavity,
    pose: "stored",
  }).hulls;

  hulls.forEach((hull, i) => {
    const desc = colliderFromHull(rapier, hull, i);
    colliders.push(desc);

    if (hull.boneIndex !== null) {
      const arr = boneMap.get(hull.boneIndex) ?? [];
      arr.push(desc);
      boneMap.set(hull.boneIndex, arr);
    }
  });

  return { colliders, boneMap };
}

export function addToWorld(
  rapier: typeof RAPIER,
  world: RAPIER.World,
  phys: PhysFile,
  position?: { x: number; y: number; z: number },
  opts?: AddToWorldOptions,
): RAPIER.RigidBody {
  if (position !== undefined && opts?.transform?.translation !== undefined) {
    throw new Error(
      "addToWorld: pass either the legacy position argument or transform.translation, not both",
    );
  }
  const transform = resolvePhysTransform({
    ...opts?.transform,
    translation:
      opts?.transform?.translation ??
      (position === undefined
        ? undefined
        : [position.x, position.y, position.z]),
  });
  const geometry = materializePhysHulls(phys, {
    lodConcavity: opts?.lodConcavity,
    pose: "bind",
  });

  // Build every descriptor before touching the world. A hull that can't be
  // placed throws, and a throw partway through attachment would otherwise
  // strand a half-populated rigid body in the simulation.
  //
  // Rigged hulls are stored bone-local; place each at its bone's bind pose so a
  // single fixed body carries the whole asset in its rest pose (rather than
  // collapsing every bone's hulls onto the body origin). The bind pose is the
  // asset's rest pose, so a single fixed body is the right home for it.
  const reflected =
    transform.scale[0] * transform.scale[1] * transform.scale[2] < 0;
  const descs = geometry.hulls.map((hull, i) => {
    const vertices = transformHullVertices(hull.vertices, {
      scale: transform.scale,
    });
    const indices = reflected
      ? reverseTriangleWinding(hull.indices)
      : hull.indices;
    return colliderFromVertices(rapier, vertices, indices, i);
  });

  const bodyDesc = rapier.RigidBodyDesc.fixed()
    .setTranslation(
      transform.translation[0],
      transform.translation[1],
      transform.translation[2],
    )
    .setRotation({
      x: transform.rotation[0],
      y: transform.rotation[1],
      z: transform.rotation[2],
      w: transform.rotation[3],
    });
  const body = world.createRigidBody(bodyDesc);
  try {
    for (const desc of descs) {
      world.createCollider(desc, body);
    }
  } catch (err) {
    // Rapier can still reject a descriptor at attach time; don't leave the
    // body behind when it does.
    world.removeRigidBody(body);
    throw err;
  }

  return body;
}

function colliderFromHull(
  rapier: typeof RAPIER,
  hull: PhysHull,
  index: number,
): RAPIER.ColliderDesc {
  return colliderFromVertices(rapier, hull.vertices, hull.indices, index);
}

function reverseTriangleWinding(indices: Uint16Array): Uint16Array {
  const reversed = indices.slice();
  for (let i = 0; i < reversed.length; i += 3) {
    const second = reversed[i + 1];
    reversed[i + 1] = reversed[i + 2];
    reversed[i + 2] = second;
  }
  return reversed;
}

function colliderFromVertices(
  rapier: typeof RAPIER,
  vertices: Float32Array,
  indices: Uint16Array,
  index: number,
): RAPIER.ColliderDesc {
  if (vertices.length < 12 || indices.length < 3) {
    throw new Error(
      `hull ${index}: too few vertices/indices to form a collider ` +
        `(${vertices.length / 3} verts, ${indices.length / 3} tris)`,
    );
  }
  // convexMesh keeps the compiled hull's own faces; convexHull would discard the
  // indices and make Rapier recompute the hull from the point cloud.
  const desc = rapier.ColliderDesc.convexMesh(
    vertices,
    Uint32Array.from(indices),
  );
  if (!desc) {
    throw new Error(
      `hull ${index}: Rapier rejected the convex mesh as degenerate`,
    );
  }
  return desc;
}
