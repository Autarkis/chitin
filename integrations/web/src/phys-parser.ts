const MAGIC = 0x53594850; // "PHYS" little-endian
const HEADER_SIZE = 32;
const FLAG_HAS_BONES = 0x01;
const FLAG_HAS_BIND_POSES = 0x02;

export interface PhysHull {
  vertices: Float32Array; // (N*3) dequantized xyz
  indices: Uint16Array;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  boneIndex: number | null;
}

export interface PhysBone {
  name: string;
  bindTransform: Float32Array; // 16 floats, row-major 4x4
}

export interface PhysFile {
  version: number;
  flags: number;
  hulls: PhysHull[];
  bones: PhysBone[];
  hasBones: boolean;
  hasBindPoses: boolean;
}

export function parsePhys(buffer: ArrayBuffer): PhysFile {
  const view = new DataView(buffer);

  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`bad magic: 0x${magic.toString(16)}`);
  }

  const version = view.getUint16(4, true);
  const flags = view.getUint16(6, true);
  const hullCount = view.getUint32(8, true);
  const totalVerts = view.getUint32(12, true);
  const totalIdx = view.getUint32(16, true);
  const hullTableOff = view.getUint32(20, true);
  const vertexDataOff = view.getUint32(24, true);
  const indexDataOff = view.getUint32(28, true);

  const hasBones = (flags & FLAG_HAS_BONES) !== 0;
  const hasBindPoses = (flags & FLAG_HAS_BIND_POSES) !== 0;
  const descSize = hasBones ? 44 : 40;

  const hulls: PhysHull[] = [];

  for (let i = 0; i < hullCount; i++) {
    const off = hullTableOff + i * descSize;
    const vOff = view.getUint32(off, true);
    const vCount = view.getUint32(off + 4, true);
    const iOff = view.getUint32(off + 8, true);
    const iCount = view.getUint32(off + 12, true);

    const aabbMin: [number, number, number] = [
      view.getFloat32(off + 16, true),
      view.getFloat32(off + 20, true),
      view.getFloat32(off + 24, true),
    ];
    const aabbMax: [number, number, number] = [
      view.getFloat32(off + 28, true),
      view.getFloat32(off + 32, true),
      view.getFloat32(off + 36, true),
    ];

    let boneIndex: number | null = null;
    if (hasBones) {
      const raw = view.getInt32(off + 40, true);
      boneIndex = raw === -1 ? null : raw;
    }

    const vertices = new Float32Array(vCount * 3);
    const qOff = vertexDataOff + vOff * 6;
    for (let v = 0; v < vCount; v++) {
      for (let c = 0; c < 3; c++) {
        const q = view.getInt16(qOff + (v * 3 + c) * 2, true);
        const extent = aabbMax[c] - aabbMin[c];
        const e = extent === 0 ? 1.0 : extent;
        vertices[v * 3 + c] = ((q + 32768) / 65535) * e + aabbMin[c];
      }
    }

    const idxOff = indexDataOff + iOff * 2;
    const indices = new Uint16Array(iCount);
    for (let t = 0; t < iCount; t++) {
      indices[t] = view.getUint16(idxOff + t * 2, true);
    }

    hulls.push({ vertices, indices, aabbMin, aabbMax, boneIndex });
  }

  const bones: PhysBone[] = [];
  if (hasBindPoses) {
    let bOff = indexDataOff + totalIdx * 2;
    const boneCount = view.getUint32(bOff, true);
    bOff += 4;
    const decoder = new TextDecoder("utf-8");

    for (let b = 0; b < boneCount; b++) {
      const bindTransform = new Float32Array(16);
      for (let f = 0; f < 16; f++) {
        bindTransform[f] = view.getFloat32(bOff + f * 4, true);
      }
      bOff += 64;
      const nameLen = view.getUint16(bOff, true);
      bOff += 2;
      const name = decoder.decode(new Uint8Array(buffer, bOff, nameLen));
      bOff += nameLen;
      bones.push({ name, bindTransform });
    }
  }

  return { version, flags, hulls, bones, hasBones, hasBindPoses };
}
