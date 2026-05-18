import {
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  Uint16BufferAttribute,
} from "three";
import type { PhysFile } from "./phys-parser.js";

export interface DebugOptions {
  color?: number;
  opacity?: number;
  wireframe?: boolean;
}

export function createDebugMeshes(
  phys: PhysFile,
  options?: DebugOptions
): Group {
  const color = options?.color ?? 0x00ff88;
  const opacity = options?.opacity ?? 0.3;
  const wireframe = options?.wireframe ?? true;

  const group = new Group();
  group.name = "chitin_colliders";

  const boneTransforms = new Map<number, Matrix4>();
  if (phys.bones.length > 0) {
    for (let i = 0; i < phys.bones.length; i++) {
      const m = new Matrix4();
      m.fromArray(phys.bones[i].bindTransform);
      boneTransforms.set(i, m);
    }
  }

  for (let i = 0; i < phys.hulls.length; i++) {
    const hull = phys.hulls[i];
    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(hull.vertices, 3));
    geo.setIndex(new Uint16BufferAttribute(hull.indices, 1));
    geo.computeVertexNormals();

    if (wireframe) {
      const edges = extractEdges(hull.indices);
      const edgeGeo = new BufferGeometry();
      edgeGeo.setAttribute(
        "position",
        new Float32BufferAttribute(hull.vertices, 3)
      );
      edgeGeo.setIndex(edges);
      const line = new LineSegments(
        edgeGeo,
        new LineBasicMaterial({ color })
      );
      line.name = `hull_${i}_wire`;
      if (hull.boneIndex !== null && boneTransforms.has(hull.boneIndex)) {
        line.applyMatrix4(boneTransforms.get(hull.boneIndex)!);
      }
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
    if (hull.boneIndex !== null && boneTransforms.has(hull.boneIndex)) {
      mesh.applyMatrix4(boneTransforms.get(hull.boneIndex)!);
    }
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
