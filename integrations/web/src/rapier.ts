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

  for (const hull of hulls) {
    const desc = colliderFromHull(rapier, hull);
    if (!desc) continue;

    colliders.push(desc);

    if (hull.boneIndex !== null) {
      const arr = boneMap.get(hull.boneIndex) ?? [];
      arr.push(desc);
      boneMap.set(hull.boneIndex, arr);
    }
  }

  return { colliders, boneMap };
}

export function addToWorld(
  rapier: typeof RAPIER,
  world: RAPIER.World,
  phys: PhysFile,
  position?: { x: number; y: number; z: number },
  opts?: ColliderOptions
): RAPIER.RigidBody {
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
  hull: PhysHull
): RAPIER.ColliderDesc | null {
  if (hull.vertices.length < 12 || hull.indices.length < 3) return null;
  return rapier.ColliderDesc.convexHull(hull.vertices);
}
