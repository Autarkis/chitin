import type RAPIER from "@dimforge/rapier3d";
import type { PhysFile, PhysHull } from "./phys-parser.js";

export interface ColliderResult {
  colliders: RAPIER.ColliderDesc[];
  boneMap: Map<number, RAPIER.ColliderDesc[]>;
}

export function createColliders(
  rapier: typeof RAPIER,
  phys: PhysFile
): ColliderResult {
  const colliders: RAPIER.ColliderDesc[] = [];
  const boneMap = new Map<number, RAPIER.ColliderDesc[]>();

  for (const hull of phys.hulls) {
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
  position?: { x: number; y: number; z: number }
): RAPIER.RigidBody {
  const pos = position ?? { x: 0, y: 0, z: 0 };
  const bodyDesc = rapier.RigidBodyDesc.fixed().setTranslation(
    pos.x,
    pos.y,
    pos.z
  );
  const body = world.createRigidBody(bodyDesc);

  const { colliders } = createColliders(rapier, phys);
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
