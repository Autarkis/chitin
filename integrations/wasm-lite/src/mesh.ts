import { ChitinError } from "./errors.js";
import { triangleCrossProduct, vertexCount } from "./geometry.js";

export interface TriangleMesh {
  vertices: Float64Array;
  faces: Int32Array;
}

export interface CanonicalizedMesh extends TriangleMesh {
  source_vertex_count: number;
  welded_vertex_count: number;
  removed_degenerate_triangles: number;
  removed_duplicate_triangles: number;
}

function coordinateKey(value: number): string {
  // Treat signed zero as the same position. All other values retain their
  // exact post-transform representation; this never merges nearby geometry.
  return Object.is(value, -0) ? "0" : String(value);
}

function vertexKey(vertices: Float64Array, offset: number): string {
  return `${coordinateKey(vertices[offset])}|${coordinateKey(vertices[offset + 1])}|${coordinateKey(vertices[offset + 2])}`;
}

function hasZeroArea(vertices: Float64Array, a: number, b: number, c: number): boolean {
  const [cx, cy, cz] = triangleCrossProduct(vertices, a, b, c);
  return cx === 0 && cy === 0 && cz === 0;
}

/**
 * Make imported render geometry safe for topology-sensitive decomposition.
 *
 * glTF commonly duplicates positions at UV and normal seams. CoACD consumes
 * geometry only, so retaining those duplicates turns a visually closed mesh
 * into disconnected faces. Exact-position welding restores the intended
 * topology without merging merely-nearby parts. Exact duplicate and zero-area
 * triangles are removed, then unused vertices are compacted.
 */
export function canonicalizeMesh(vertices: Float64Array, faces: Int32Array): CanonicalizedMesh {
  if (vertices.length % 3 !== 0 || faces.length % 3 !== 0) {
    throw new ChitinError("INVALID_MESH", "mesh arrays must contain xyz vertices and triangle faces");
  }

  const sourceVertexCount = vertexCount(vertices);
  const representativeByKey = new Map<string, number>();
  const canonicalVertices: number[] = [];
  const sourceToCanonical = new Int32Array(sourceVertexCount);
  for (let source = 0; source < sourceVertexCount; source++) {
    const offset = source * 3;
    const key = vertexKey(vertices, offset);
    let canonical = representativeByKey.get(key);
    if (canonical === undefined) {
      canonical = canonicalVertices.length / 3;
      representativeByKey.set(key, canonical);
      canonicalVertices.push(vertices[offset], vertices[offset + 1], vertices[offset + 2]);
    }
    sourceToCanonical[source] = canonical;
  }

  const weldedVertices = new Float64Array(canonicalVertices);
  const keptFaces: number[] = [];
  const uniqueFaces = new Set<string>();
  let removedDegenerate = 0;
  let removedDuplicate = 0;
  for (let offset = 0; offset < faces.length; offset += 3) {
    const sourceA = faces[offset];
    const sourceB = faces[offset + 1];
    const sourceC = faces[offset + 2];
    if (
      sourceA < 0 || sourceA >= sourceVertexCount ||
      sourceB < 0 || sourceB >= sourceVertexCount ||
      sourceC < 0 || sourceC >= sourceVertexCount
    ) {
      throw new ChitinError("INVALID_MESH", `face ${offset / 3} references a missing vertex`);
    }
    const a = sourceToCanonical[sourceA];
    const b = sourceToCanonical[sourceB];
    const c = sourceToCanonical[sourceC];
    if (a === b || b === c || c === a || hasZeroArea(weldedVertices, a, b, c)) {
      removedDegenerate++;
      continue;
    }
    const duplicateKey = [a, b, c].sort((left, right) => left - right).join("|");
    if (uniqueFaces.has(duplicateKey)) {
      removedDuplicate++;
      continue;
    }
    uniqueFaces.add(duplicateKey);
    keptFaces.push(a, b, c);
  }

  if (keptFaces.length === 0) {
    throw new ChitinError("INVALID_MESH", "mesh has no non-degenerate triangles after canonicalization");
  }

  const referenced = new Map<number, number>();
  const compactVertices: number[] = [];
  const compactFaces = new Int32Array(keptFaces.length);
  for (let index = 0; index < keptFaces.length; index++) {
    const canonical = keptFaces[index];
    let compact = referenced.get(canonical);
    if (compact === undefined) {
      compact = compactVertices.length / 3;
      referenced.set(canonical, compact);
      compactVertices.push(
        weldedVertices[canonical * 3],
        weldedVertices[canonical * 3 + 1],
        weldedVertices[canonical * 3 + 2],
      );
    }
    compactFaces[index] = compact;
  }

  return {
    vertices: new Float64Array(compactVertices),
    faces: compactFaces,
    source_vertex_count: sourceVertexCount,
    welded_vertex_count: vertexCount(compactVertices),
    removed_degenerate_triangles: removedDegenerate,
    removed_duplicate_triangles: removedDuplicate,
  };
}

/** Split a mesh into vertex-connected triangle components for CoACD. */
export function splitMeshComponents(vertices: Float64Array, faces: Int32Array): TriangleMesh[] {
  const totalVertices = vertexCount(vertices);
  const parents = Int32Array.from({ length: totalVertices }, (_, index) => index);
  const find = (start: number): number => {
    let root = start;
    while (parents[root] !== root) root = parents[root];
    let current = start;
    while (parents[current] !== current) {
      const next = parents[current];
      parents[current] = root;
      current = next;
    }
    return root;
  };
  const union = (left: number, right: number): void => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) parents[rightRoot] = leftRoot;
  };

  for (let offset = 0; offset < faces.length; offset += 3) {
    union(faces[offset], faces[offset + 1]);
    union(faces[offset], faces[offset + 2]);
  }

  const facesByRoot = new Map<number, number[]>();
  for (let offset = 0; offset < faces.length; offset += 3) {
    const root = find(faces[offset]);
    const componentFaces = facesByRoot.get(root) ?? [];
    componentFaces.push(faces[offset], faces[offset + 1], faces[offset + 2]);
    facesByRoot.set(root, componentFaces);
  }

  return Array.from(facesByRoot.values(), (componentFaces) => {
    const globalToLocal = new Map<number, number>();
    const localVertices: number[] = [];
    const localFaces = new Int32Array(componentFaces.length);
    for (let index = 0; index < componentFaces.length; index++) {
      const global = componentFaces[index];
      let local = globalToLocal.get(global);
      if (local === undefined) {
        local = localVertices.length / 3;
        globalToLocal.set(global, local);
        localVertices.push(
          vertices[global * 3],
          vertices[global * 3 + 1],
          vertices[global * 3 + 2],
        );
      }
      localFaces[index] = local;
    }
    return { vertices: new Float64Array(localVertices), faces: localFaces };
  });
}
