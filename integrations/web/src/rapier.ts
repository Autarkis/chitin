import type RAPIER from "@dimforge/rapier3d-compat";
import type { PhysFile, PhysHull } from "./phys-parser.js";
import { selectLodHulls } from "./phys-parser.js";

export interface ColliderResult {
  colliders: RAPIER.ColliderDesc[];
  boneMap: Map<number, RAPIER.ColliderDesc[]>;
}

export interface ColliderOptions {
  // Choose the LOD tier nearest this concavity. Omit for LOD 0 (highest detail).
  lodConcavity?: number;
}

export function createColliders(
  rapier: typeof RAPIER,
  phys: PhysFile,
  opts?: ColliderOptions
): ColliderResult {
  const colliders: RAPIER.ColliderDesc[] = [];
  const boneMap = new Map<number, RAPIER.ColliderDesc[]>();

  const hulls =
    opts?.lodConcavity !== undefined
      ? selectLodHulls(phys, opts.lodConcavity)
      : phys.hulls;

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
  opts?: ColliderOptions
): RAPIER.RigidBody {
  if (phys.hasBones) {
    throw new Error(
      "addToWorld cannot place a rigged .phys: its hulls are bone-local and would " +
        "collapse onto a single origin. Use createColliders(rapier, phys) and attach " +
        "each boneMap entry at its bone's bind pose."
    );
  }
  const pos = position ?? { x: 0, y: 0, z: 0 };
  const bodyDesc = rapier.RigidBodyDesc.fixed().setTranslation(
    pos.x,
    pos.y,
    pos.z
  );
  const body = world.createRigidBody(bodyDesc);

  const { colliders } = createColliders(rapier, phys, opts);
  for (const desc of colliders) {
    world.createCollider(desc, body);
  }

  return body;
}

function colliderFromHull(
  rapier: typeof RAPIER,
  hull: PhysHull,
  index: number
): RAPIER.ColliderDesc {
  if (hull.vertices.length < 12 || hull.indices.length < 3) {
    throw new Error(
      `hull ${index}: too few vertices/indices to form a collider ` +
        `(${hull.vertices.length / 3} verts, ${hull.indices.length / 3} tris)`
    );
  }
  // convexMesh keeps the compiled hull's own faces; convexHull would discard the
  // indices and make Rapier recompute the hull from the point cloud.
  const desc = rapier.ColliderDesc.convexMesh(
    hull.vertices,
    Uint32Array.from(hull.indices)
  );
  if (!desc) {
    throw new Error(
      `hull ${index}: Rapier rejected the convex mesh as degenerate`
    );
  }
  return desc;
}
