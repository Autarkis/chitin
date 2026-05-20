import type { ConvexHull, QuantizedHull } from "./types.js";

const MAGIC = 0x53594850; // "PHYS" little-endian
const VERSION = 3;
const HEADER_SIZE = 32;
const HULL_DESC_SIZE = 40; // no bones

export function quantizeHulls(hulls: ConvexHull[]): QuantizedHull[] {
  return hulls.map((hull) => {
    const nv = hull.vertices.length / 3;
    const aabbMin: [number, number, number] = [Infinity, Infinity, Infinity];
    const aabbMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < nv; i++) {
      for (let c = 0; c < 3; c++) {
        const v = hull.vertices[i * 3 + c];
        if (v < aabbMin[c]) aabbMin[c] = v;
        if (v > aabbMax[c]) aabbMax[c] = v;
      }
    }

    const quantized = new Int16Array(nv * 3);
    for (let i = 0; i < nv; i++) {
      for (let c = 0; c < 3; c++) {
        const extent = aabbMax[c] - aabbMin[c];
        const e = extent === 0 ? 1.0 : extent;
        const normalized = (hull.vertices[i * 3 + c] - aabbMin[c]) / e;
        quantized[i * 3 + c] = Math.max(
          -32768,
          Math.min(32767, Math.round(normalized * 65535 - 32768)),
        );
      }
    }

    const indices = new Uint16Array(hull.indices.length);
    for (let i = 0; i < hull.indices.length; i++) {
      indices[i] = hull.indices[i];
    }

    return { aabbMin, aabbMax, quantizedVertices: quantized, indices };
  });
}

export function writePhys(hulls: ConvexHull[]): ArrayBuffer {
  const quantized = quantizeHulls(hulls);

  let totalVerts = 0;
  let totalIdx = 0;
  for (const h of quantized) {
    totalVerts += h.quantizedVertices.length / 3;
    totalIdx += h.indices.length;
  }

  const hullTableSize = quantized.length * HULL_DESC_SIZE;
  const vertexDataSize = totalVerts * 6; // int16 * 3 per vert
  const indexDataSize = totalIdx * 2; // uint16 per index

  const hullTableOff = HEADER_SIZE;
  const vertexDataOff = hullTableOff + hullTableSize;
  const indexDataOff = vertexDataOff + vertexDataSize;
  const totalSize = indexDataOff + indexDataSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // Header
  view.setUint32(0, MAGIC, true);
  view.setUint16(4, VERSION, true);
  view.setUint16(6, 0, true); // flags: no bones, no bind poses, no LOD
  view.setUint32(8, quantized.length, true);
  view.setUint32(12, totalVerts, true);
  view.setUint32(16, totalIdx, true);
  view.setUint32(20, hullTableOff, true);
  view.setUint32(24, vertexDataOff, true);
  view.setUint32(28, indexDataOff, true);

  // Hull descriptors + vertex/index data
  let vertOff = 0;
  let idxOff = 0;

  for (let i = 0; i < quantized.length; i++) {
    const h = quantized[i];
    const nv = h.quantizedVertices.length / 3;
    const ni = h.indices.length;
    const descOff = hullTableOff + i * HULL_DESC_SIZE;

    view.setUint32(descOff, vertOff, true);
    view.setUint32(descOff + 4, nv, true);
    view.setUint32(descOff + 8, idxOff, true);
    view.setUint32(descOff + 12, ni, true);
    view.setFloat32(descOff + 16, h.aabbMin[0], true);
    view.setFloat32(descOff + 20, h.aabbMin[1], true);
    view.setFloat32(descOff + 24, h.aabbMin[2], true);
    view.setFloat32(descOff + 28, h.aabbMax[0], true);
    view.setFloat32(descOff + 32, h.aabbMax[1], true);
    view.setFloat32(descOff + 36, h.aabbMax[2], true);

    // Vertex data
    for (let v = 0; v < nv * 3; v++) {
      view.setInt16(vertexDataOff + (vertOff * 3 + v) * 2, h.quantizedVertices[v], true);
    }

    // Index data
    for (let t = 0; t < ni; t++) {
      view.setUint16(indexDataOff + (idxOff + t) * 2, h.indices[t], true);
    }

    vertOff += nv;
    idxOff += ni;
  }

  return buffer;
}
