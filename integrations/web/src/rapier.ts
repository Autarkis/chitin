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
  const pos = position ?? { x: 0, y: 0, z: 0 };
  const bodyDesc = rapier.RigidBodyDesc.fixed().setTranslation(
    pos.x,
    pos.y,
    pos.z
  );
  const body = world.createRigidBody(bodyDesc);

  const hulls =
    opts?.lodConcavity !== undefined
      ? selectLodHulls(phys, opts.lodConcavity)
      : phys.hulls;

  // Rigged hulls are stored bone-local; place each at its bone's bind pose so a
  // single fixed body carries the whole asset in its rest pose (rather than
  // collapsing every bone's hulls onto the body origin). The bind pose is the
  // asset's rest pose, so a single fixed body is the right home for it.
  hulls.forEach((hull, i) => {
    let vertices = hull.vertices;
    if (hull.boneIndex !== null) {
      if (!phys.hasBindPoses) {
        throw new Error(
          `addToWorld: hull ${i} is bone-local (bone ${hull.boneIndex}) but the ` +
            `.phys carries no bind poses to place it. Recompile with bind poses, or ` +
            `use createColliders() and attach each boneMap entry at its own pose.`
        );
      }
      // The parser range-checks boneIndex against the bone table whenever bind
      // poses are present, so this lookup is always in bounds here.
      vertices = applyBindPose(hull.vertices, phys.bones[hull.boneIndex].bindTransform);
    }
    const desc = colliderFromVertices(rapier, vertices, hull.indices, i);
    world.createCollider(desc, body);
  });

  return body;
}

// Reconstruct a rigged hull's world-space (model-space) vertices from its
// bone-local vertices and the bone's bind transform, matching the .phys
// contract `world = local @ bind_transform` where `bind_transform` is a
// row-major 4x4 and `local` is a row vector (docs/phys.md, "Bind Pose Block").
// Baking the full affine into the vertices — rather than pushing it onto the
// collider — keeps any scale/shear in the matrix exact, which a Rapier collider
// pose (translation + rotation only) could not represent.
export function applyBindPose(
  vertices: Float32Array,
  bindTransform: Float32Array
): Float32Array {
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

function colliderFromHull(
  rapier: typeof RAPIER,
  hull: PhysHull,
  index: number
): RAPIER.ColliderDesc {
  return colliderFromVertices(rapier, hull.vertices, hull.indices, index);
}

function colliderFromVertices(
  rapier: typeof RAPIER,
  vertices: Float32Array,
  indices: Uint16Array,
  index: number
): RAPIER.ColliderDesc {
  if (vertices.length < 12 || indices.length < 3) {
    throw new Error(
      `hull ${index}: too few vertices/indices to form a collider ` +
        `(${vertices.length / 3} verts, ${indices.length / 3} tris)`
    );
  }
  // convexMesh keeps the compiled hull's own faces; convexHull would discard the
  // indices and make Rapier recompute the hull from the point cloud.
  const desc = rapier.ColliderDesc.convexMesh(
    vertices,
    Uint32Array.from(indices)
  );
  if (!desc) {
    throw new Error(
      `hull ${index}: Rapier rejected the convex mesh as degenerate`
    );
  }
  return desc;
}
