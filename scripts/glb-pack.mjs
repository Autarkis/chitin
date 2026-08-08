const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

function padded(length) {
  return (length + 3) & ~3;
}

export function packGlb(document, binary, declaredBinaryLength) {
  const source = binary instanceof ArrayBuffer
    ? new Uint8Array(binary)
    : new Uint8Array(binary.buffer, binary.byteOffset, binary.byteLength);
  const contentLength = declaredBinaryLength ?? source.byteLength;
  if (contentLength > source.byteLength) {
    throw new RangeError("declared GLB binary length exceeds the supplied data");
  }

  const jsonBytes = new TextEncoder().encode(JSON.stringify(document));
  const jsonLength = padded(jsonBytes.byteLength);
  const binaryLength = padded(contentLength);
  const total = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new ArrayBuffer(total);
  const view = new DataView(output);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, JSON_CHUNK, true);
  const jsonOutput = new Uint8Array(output, 20, jsonLength);
  jsonOutput.fill(0x20);
  jsonOutput.set(jsonBytes);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binaryLength, true);
  view.setUint32(binaryHeader + 4, BIN_CHUNK, true);
  new Uint8Array(output, binaryHeader + 8, contentLength).set(
    source.subarray(0, contentLength),
  );
  return output;
}

function computePositionBounds(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], positions[i + axis]);
      max[axis] = Math.max(max[axis], positions[i + axis]);
    }
  }
  return { min, max };
}

// Builds the smallest glTF document + interleaved binary buffer that can
// carry a flat position/index mesh (no normals, materials, or skinning).
// Callers pass the returned { document, binary } straight into packGlb.
export function buildMinimalGltf(positions, indices, options = {}) {
  const { generator = "chitin minimal glTF envelope", mode, bounds = true, min, max } = options;

  let indexComponentType;
  if (indices instanceof Uint16Array) {
    indexComponentType = 5123;
  } else if (indices instanceof Uint32Array) {
    indexComponentType = 5125;
  } else {
    throw new TypeError("indices must be a Uint16Array or Uint32Array");
  }

  const positionBytes = positions.byteLength;
  const indexOffset = (positionBytes + 3) & ~3;
  const binary = new Uint8Array(indexOffset + indices.byteLength);
  binary.set(new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength), 0);
  binary.set(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength), indexOffset);

  const positionAccessor = {
    bufferView: 0,
    componentType: 5126,
    count: positions.length / 3,
    type: "VEC3",
  };
  if (bounds) {
    const computed = min && max ? { min, max } : computePositionBounds(positions);
    positionAccessor.min = computed.min;
    positionAccessor.max = computed.max;
  }

  const primitive = { attributes: { POSITION: 0 }, indices: 1 };
  if (mode !== undefined) primitive.mode = mode;

  const document = {
    asset: { version: "2.0", generator },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [primitive] }],
    accessors: [
      positionAccessor,
      { bufferView: 1, componentType: indexComponentType, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes, target: 34962 },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength, target: 34963 },
    ],
    buffers: [{ byteLength: indexOffset + indices.byteLength }],
  };

  return { document, binary };
}
