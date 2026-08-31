import type { BufferGeometry, Object3D, Mesh as ThreeMesh } from "three";
import { ChitinError } from "./errors.js";

export interface ExtractedMesh {
  vertices: Float64Array;
  faces: Int32Array;
}

/**
 * Extract a triangle mesh from a Three.js BufferGeometry.
 *
 * The geometry must have a `position` attribute. Indexed and non-indexed
 * geometries are both supported; non-triangle draw modes are rejected.
 */
export function geometryToMesh(geometry: BufferGeometry): ExtractedMesh {
  const position = geometry.getAttribute("position");
  if (!position) {
    throw new ChitinError("INVALID_MESH", "BufferGeometry has no position attribute", {
      stage: "parsing-input",
      suggestion: "Provide a geometry with a float position attribute.",
    });
  }

  const vertexCount = position.count;
  const index = geometry.getIndex();

  let faceCount: number;
  let faces: Int32Array;

  if (index) {
    if (index.count % 3 !== 0) {
      throw new ChitinError("INVALID_MESH", `Index count ${index.count} is not a multiple of 3`, {
        stage: "parsing-input",
        suggestion: "Provide triangle-list geometry.",
      });
    }
    faceCount = index.count / 3;
    faces = new Int32Array(index.count);
    for (let i = 0; i < index.count; i++) {
      const vertexIndex = index.getX(i);
      if (!Number.isInteger(vertexIndex) || vertexIndex < 0 || vertexIndex >= vertexCount) {
        throw new ChitinError(
          "INVALID_MESH",
          `Index ${vertexIndex} at offset ${i} is outside the vertex range [0, ${vertexCount})`,
          {
            stage: "parsing-input",
            suggestion: "Provide geometry whose indices reference existing vertices.",
          },
        );
      }
      faces[i] = vertexIndex;
    }
  } else {
    if (vertexCount % 3 !== 0) {
      throw new ChitinError(
        "INVALID_MESH",
        `Non-indexed vertex count ${vertexCount} is not a multiple of 3`,
        {
          stage: "parsing-input",
          suggestion: "Provide triangle-list geometry or add an index.",
        },
      );
    }
    faceCount = vertexCount / 3;
    faces = new Int32Array(faceCount * 3);
    for (let i = 0; i < faces.length; i++) {
      faces[i] = i;
    }
  }

  const vertices = new Float64Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    vertices[i * 3] = position.getX(i);
    vertices[i * 3 + 1] = position.getY(i);
    vertices[i * 3 + 2] = position.getZ(i);
  }

  return { vertices, faces };
}

/**
 * Collect and merge all Mesh geometries under an Object3D, applying world
 * transforms. Returns a single merged triangle mesh suitable for compilation.
 */
export function collectMeshes(root: Object3D): ExtractedMesh {
  root.updateMatrixWorld(true);

  const allVertices: number[] = [];
  const allFaces: number[] = [];
  let vertexOffset = 0;

  root.traverse((child) => {
    if (!isMesh(child)) return;
    const mesh = child as ThreeMesh;
    const geometry = mesh.geometry;
    const position = geometry.getAttribute("position");
    if (!position) return;
    const extracted = geometryToMesh(geometry);

    const matrix = mesh.matrixWorld;
    const vertexCount = extracted.vertices.length / 3;

    for (let i = 0; i < vertexCount; i++) {
      const x = extracted.vertices[i * 3];
      const y = extracted.vertices[i * 3 + 1];
      const z = extracted.vertices[i * 3 + 2];

      const e = matrix.elements;
      const tx = e[0] * x + e[4] * y + e[8] * z + e[12];
      const ty = e[1] * x + e[5] * y + e[9] * z + e[13];
      const tz = e[2] * x + e[6] * y + e[10] * z + e[14];

      allVertices.push(tx, ty, tz);
    }

    for (const face of extracted.faces) {
      allFaces.push(face + vertexOffset);
    }

    vertexOffset += vertexCount;
  });

  if (allFaces.length === 0) {
    throw new ChitinError("INVALID_MESH", "No mesh geometry found in the scene graph", {
      stage: "parsing-input",
      suggestion: "Provide an Object3D containing at least one Mesh with geometry.",
    });
  }

  return {
    vertices: new Float64Array(allVertices),
    faces: new Int32Array(allFaces),
  };
}

function isMesh(object: Object3D): boolean {
  return (object as ThreeMesh).isMesh === true;
}
