const MAGIC = 0x53594850; // "PHYS" little-endian
const HEADER_SIZE = 32;
const FLAG_HAS_BONES = 0x01;
const FLAG_HAS_BIND_POSES = 0x02;
const FLAG_HAS_LOD = 0x04;
const KNOWN_FLAGS = FLAG_HAS_BONES | FLAG_HAS_BIND_POSES | FLAG_HAS_LOD;
const LOD_TIER_HEADER_SIZE = 24;
const SUPPORTED_VERSIONS = new Set([2, 3]);

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
  hasLod: boolean;
}

export function parsePhys(buffer: ArrayBuffer): PhysFile {
  const view = new DataView(buffer);
  const byteLength = buffer.byteLength;

  if (byteLength < HEADER_SIZE) {
    throw new Error(`file too small: ${byteLength} bytes`);
  }

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

  if (!SUPPORTED_VERSIONS.has(version)) {
    throw new Error(`unsupported .phys version ${version}`);
  }
  const unknownFlags = flags & ~KNOWN_FLAGS;
  if (unknownFlags !== 0) {
    throw new Error(`unknown flags 0x${unknownFlags.toString(16)}`);
  }

  const hasBones = (flags & FLAG_HAS_BONES) !== 0;
  const hasBindPoses = (flags & FLAG_HAS_BIND_POSES) !== 0;
  const hasLod = (flags & FLAG_HAS_LOD) !== 0;
  const descSize = hasBones ? 44 : 40;

  const expectedVertexOff = hullTableOff + hullCount * descSize;
  const expectedIndexOff = vertexDataOff + totalVerts * 6;
  if (hullTableOff !== HEADER_SIZE) {
    throw new Error(`bad hull table offset: ${hullTableOff}`);
  }
  if (vertexDataOff !== expectedVertexOff) {
    throw new Error(`bad vertex data offset: ${vertexDataOff}`);
  }
  if (indexDataOff !== expectedIndexOff) {
    throw new Error(`bad index data offset: ${indexDataOff}`);
  }

  // Peek the bone count (the bind-pose block trails the index data) so each
  // hull's boneIndex can be range-checked in the loop below.
  let boneTableCount: number | null = null;
  if (hasBindPoses) {
    const boneBlockOff = indexDataOff + totalIdx * 2;
    if (boneBlockOff + 4 <= byteLength) {
      boneTableCount = view.getUint32(boneBlockOff, true);
    }
  }

  const hulls: PhysHull[] = [];
  let expectedVOff = 0;
  let expectedIOff = 0;

  for (let i = 0; i < hullCount; i++) {
    const off = hullTableOff + i * descSize;
    requireBytes(byteLength, off, descSize, `hull ${i} descriptor`);
    const vOff = view.getUint32(off, true);
    const vCount = view.getUint32(off + 4, true);
    const iOff = view.getUint32(off + 8, true);
    const iCount = view.getUint32(off + 12, true);

    // Each hull's declared vertex/index range must stay within the arrays.
    if (vOff + vCount > totalVerts) {
      throw new Error(
        `hull ${i}: vertex range [${vOff}, ${vOff + vCount}) exceeds total_vertices ${totalVerts}`,
      );
    }
    if (iOff + iCount > totalIdx) {
      throw new Error(
        `hull ${i}: index range [${iOff}, ${iOff + iCount}) exceeds total_indices ${totalIdx}`,
      );
    }

    // Hull ranges are contiguous and non-overlapping: each offset must equal
    // the running total of preceding counts.
    if (vOff !== expectedVOff) {
      throw new Error(
        `hull ${i}: vertex_offset ${vOff} != expected ${expectedVOff} (non-contiguous or overlapping range)`,
      );
    }
    if (iOff !== expectedIOff) {
      throw new Error(
        `hull ${i}: index_offset ${iOff} != expected ${expectedIOff} (non-contiguous or overlapping range)`,
      );
    }

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
    if (![...aabbMin, ...aabbMax].every(Number.isFinite)) {
      throw new Error(`hull ${i}: non-finite aabb`);
    }

    let boneIndex: number | null = null;
    if (hasBones) {
      const raw = view.getInt32(off + 40, true);
      if (raw < -1) {
        throw new Error(`hull ${i}: invalid bone_index ${raw}`);
      }
      if (boneTableCount !== null && raw >= boneTableCount) {
        throw new Error(
          `hull ${i}: bone_index ${raw} >= bone_count ${boneTableCount}`,
        );
      }
      boneIndex = raw === -1 ? null : raw;
    }

    const vertices = new Float32Array(vCount * 3);
    const qOff = vertexDataOff + vOff * 6;
    requireBytes(byteLength, qOff, vCount * 6, `hull ${i} vertices`);
    for (let v = 0; v < vCount; v++) {
      for (let c = 0; c < 3; c++) {
        const q = view.getInt16(qOff + (v * 3 + c) * 2, true);
        const extent = aabbMax[c] - aabbMin[c];
        const e = extent === 0 ? 1.0 : extent;
        vertices[v * 3 + c] = ((q + 32768) / 65535) * e + aabbMin[c];
      }
    }

    const idxOff = indexDataOff + iOff * 2;
    requireBytes(byteLength, idxOff, iCount * 2, `hull ${i} indices`);
    const indices = new Uint16Array(iCount);
    for (let t = 0; t < iCount; t++) {
      indices[t] = view.getUint16(idxOff + t * 2, true);
    }

    hulls.push({ vertices, indices, aabbMin, aabbMax, boneIndex });
    expectedVOff += vCount;
    expectedIOff += iCount;
  }

  const bones: PhysBone[] = [];
  let nextBlockOff = indexDataOff + totalIdx * 2;
  if (hasBindPoses) {
    let bOff = nextBlockOff;
    requireBytes(byteLength, bOff, 4, "bone count");
    const boneCount = view.getUint32(bOff, true);
    bOff += 4;
    const decoder = new TextDecoder("utf-8");

    for (let b = 0; b < boneCount; b++) {
      requireBytes(byteLength, bOff, 64, `bone ${b} bind transform`);
      const bindTransform = new Float32Array(16);
      for (let f = 0; f < 16; f++) {
        bindTransform[f] = view.getFloat32(bOff + f * 4, true);
      }
      if (!bindTransform.every(Number.isFinite)) {
        throw new Error(`bone ${b}: non-finite bind_transform`);
      }
      bOff += 64;
      requireBytes(byteLength, bOff, 2, `bone ${b} name length`);
      const nameLen = view.getUint16(bOff, true);
      bOff += 2;
      requireBytes(byteLength, bOff, nameLen, `bone ${b} name`);
      const name = decoder.decode(new Uint8Array(buffer, bOff, nameLen));
      bOff += nameLen;
      bones.push({ name, bindTransform });
    }
    nextBlockOff = bOff;
  }

  if (hasLod) {
    requireBytes(byteLength, nextBlockOff, 4, "LOD tier count");
    const tierCount = view.getUint32(nextBlockOff, true);
    nextBlockOff += 4;
    for (let tier = 0; tier < tierCount; tier++) {
      requireBytes(
        byteLength,
        nextBlockOff,
        LOD_TIER_HEADER_SIZE,
        `LOD tier ${tier} header`
      );
      const dataSize = view.getUint32(nextBlockOff + 16, true);
      nextBlockOff += LOD_TIER_HEADER_SIZE;
      requireBytes(byteLength, nextBlockOff, dataSize, `LOD tier ${tier} data`);
      nextBlockOff += dataSize;
    }
  }

  if (byteLength !== nextBlockOff) {
    throw new Error(`${byteLength - nextBlockOff} trailing bytes after end of data`);
  }

  return { version, flags, hulls, bones, hasBones, hasBindPoses, hasLod };
}

function requireBytes(
  byteLength: number,
  offset: number,
  size: number,
  label: string
): void {
  if (offset < 0 || size < 0 || offset + size > byteLength) {
    throw new Error(`${label} truncated`);
  }
}
