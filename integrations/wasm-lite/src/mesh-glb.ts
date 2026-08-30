import type { TriangleMesh } from "./mesh.js";

export function encodeTriangleMeshGlb(mesh: TriangleMesh): ArrayBuffer {
  const positions = Float32Array.from(mesh.vertices);
  const indices = Uint32Array.from(mesh.faces);
  const positionBytes = positions.byteLength;
  const indexOffset = (positionBytes + 3) & ~3;
  const binaryLength = indexOffset + indices.byteLength;
  const binary = new Uint8Array(binaryLength);
  binary.set(new Uint8Array(positions.buffer), 0);
  binary.set(new Uint8Array(indices.buffer), indexOffset);
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < positions.length; offset += 3) {
    for (let axis = 0; axis < 3; axis++) {
      mins[axis] = Math.min(mins[axis], positions[offset + axis]);
      maxs[axis] = Math.max(maxs[axis], positions[offset + axis]);
    }
  }
  const document = {
    asset: { version: "2.0", generator: "chitin poisson reconstruction" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: "VEC3", min: mins, max: maxs },
      { bufferView: 1, componentType: 5125, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes, target: 34962 },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength, target: 34963 },
    ],
    buffers: [{ byteLength: binaryLength }],
  };
  const encoder = new TextEncoder();
  const json = encoder.encode(JSON.stringify(document));
  const jsonLength = (json.length + 3) & ~3;
  const totalLength = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new ArrayBuffer(totalLength);
  const view = new DataView(output);
  const bytes = new Uint8Array(output);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(json, 20);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binaryLength, true);
  view.setUint32(binaryHeader + 4, 0x004e4942, true);
  bytes.set(binary, binaryHeader + 8);
  return output;
}

export function autoShellThickness(vertices: Float64Array): number {
  const n = vertices.length / 3;
  if (n === 0) return 0.01;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = vertices[i * 3], y = vertices[i * 3 + 1], z = vertices[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const extents = [maxX - minX, maxY - minY, maxZ - minZ].sort((a, b) => a - b);
  return extents[1] * 0.02;
}
