export interface FixtureOptions {
  nodeScale?: [number, number, number];
  primitiveMode?: number;
  externalBuffer?: boolean;
  morphTargets?: boolean;
}

function padded(length: number): number {
  return (length + 3) & ~3;
}

export function packGlb(doc: object, binary: ArrayBuffer, declaredBinaryLength = binary.byteLength): ArrayBuffer {
  const jsonText = JSON.stringify(doc);
  const jsonBytes = new TextEncoder().encode(jsonText);
  const jsonLength = padded(jsonBytes.length);
  const binLength = padded(declaredBinaryLength);
  const total = 12 + 8 + jsonLength + 8 + binLength;
  const output = new ArrayBuffer(total);
  const view = new DataView(output);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  const jsonOutput = new Uint8Array(output, 20, jsonLength);
  jsonOutput.fill(0x20);
  jsonOutput.set(jsonBytes);
  const binHeader = 20 + jsonLength;
  view.setUint32(binHeader, binLength, true);
  view.setUint32(binHeader + 4, 0x004e4942, true);
  new Uint8Array(output, binHeader + 8, binLength).set(
    new Uint8Array(binary, 0, declaredBinaryLength),
  );
  return output;
}

export function makeGlb(options: FixtureOptions = {}): ArrayBuffer {
  // Interleaved xyz + one ignored float proves byteStride handling.
  const binary = new ArrayBuffer(56);
  const data = new DataView(binary);
  const positions = [0, 0, 0, 99, 1, 0, 0, 99, 0, 1, 0, 99];
  positions.forEach((value, index) => data.setFloat32(index * 4, value, true));
  data.setUint16(48, 0, true);
  data.setUint16(50, 1, true);
  data.setUint16(52, 2, true);

  return packGlb(
    {
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [0, 1] }],
      nodes: [
        { mesh: 0, scale: options.nodeScale },
        { mesh: 0, translation: [10, 0, 0] },
      ],
      meshes: [
        {
          primitives: [
            {
              attributes: { POSITION: 0 },
              indices: 1,
              mode: options.primitiveMode,
              targets: options.morphTargets ? [{}] : undefined,
            },
            { attributes: { POSITION: 0 } },
          ],
        },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
        { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 48, byteStride: 16 },
        { buffer: 0, byteOffset: 48, byteLength: 6 },
      ],
      buffers: [
        options.externalBuffer
          ? { byteLength: 54, uri: "geometry.bin" }
          : { byteLength: 54 },
      ],
    },
    binary,
    54,
  );
}

export function makeSparseGlb(): ArrayBuffer {
  const binary = new ArrayBuffer(40);
  new Uint8Array(binary, 0, 3).set([0, 1, 2]);
  new Float32Array(binary, 4, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return packGlb(
    {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [
        {
          componentType: 5126,
          count: 3,
          type: "VEC3",
          sparse: {
            count: 3,
            indices: { bufferView: 0, componentType: 5121 },
            values: { bufferView: 1 },
          },
        },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 3 },
        { buffer: 0, byteOffset: 4, byteLength: 36 },
      ],
      buffers: [{ byteLength: 40 }],
    },
    binary,
  );
}

/** Closed, thin-walled tray whose free interior must not be bridged by coarse hulls. */
export function makeThinOpenTrayGlb(): ArrayBuffer {
  const vertices = new Float32Array([
    -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0,
    -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
    -0.99, -0.99, 0.01, 0.99, -0.99, 0.01, 0.99, 0.99, 0.01, -0.99, 0.99, 0.01,
    -0.99, -0.99, 1, 0.99, -0.99, 1, 0.99, 0.99, 1, -0.99, 0.99, 1,
  ]);
  const faces: number[] = [];
  const quad = (a: number, b: number, c: number, d: number) => {
    faces.push(a, b, c, a, c, d);
  };
  quad(0, 3, 2, 1);
  quad(0, 1, 5, 4);
  quad(1, 2, 6, 5);
  quad(2, 3, 7, 6);
  quad(3, 0, 4, 7);
  quad(8, 9, 10, 11);
  quad(8, 12, 13, 9);
  quad(9, 13, 14, 10);
  quad(10, 14, 15, 11);
  quad(11, 15, 12, 8);
  quad(4, 5, 13, 12);
  quad(5, 6, 14, 13);
  quad(6, 7, 15, 14);
  quad(7, 4, 12, 15);
  const indices = new Uint16Array(faces);
  const indexOffset = vertices.byteLength;
  const binary = new ArrayBuffer(indexOffset + indices.byteLength);
  new Float32Array(binary, 0, vertices.length).set(vertices);
  new Uint16Array(binary, indexOffset, indices.length).set(indices);
  return packGlb({
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: vertices.length / 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: vertices.byteLength },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength },
    ],
    buffers: [{ byteLength: binary.byteLength }],
  }, binary);
}

/** Three equally tessellated solids isolate roundness and scene-size budgeting. */
export function makeAdaptiveHullBudgetGlb(): ArrayBuffer {
  const latitudeBands = 10;
  const longitudeBands = 20;
  const points: number[][] = [[0, 1, 0]];
  for (let latitude = 1; latitude < latitudeBands; latitude++) {
    const phi = (Math.PI * latitude) / latitudeBands;
    for (let longitude = 0; longitude < longitudeBands; longitude++) {
      const theta = (2 * Math.PI * longitude) / longitudeBands;
      points.push([
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      ]);
    }
  }
  const bottom = points.length;
  points.push([0, -1, 0]);
  const triangles: number[][] = [];
  for (let longitude = 0; longitude < longitudeBands; longitude++) {
    const next = (longitude + 1) % longitudeBands;
    triangles.push([0, 1 + next, 1 + longitude]);
  }
  for (let latitude = 0; latitude < latitudeBands - 2; latitude++) {
    const current = 1 + latitude * longitudeBands;
    const nextRing = current + longitudeBands;
    for (let longitude = 0; longitude < longitudeBands; longitude++) {
      const next = (longitude + 1) % longitudeBands;
      triangles.push(
        [current + longitude, current + next, nextRing + next],
        [current + longitude, nextRing + next, nextRing + longitude],
      );
    }
  }
  const lastRing = 1 + (latitudeBands - 2) * longitudeBands;
  for (let longitude = 0; longitude < longitudeBands; longitude++) {
    const next = (longitude + 1) % longitudeBands;
    triangles.push([lastRing + longitude, lastRing + next, bottom]);
  }

  const vertices = new Float32Array(points.flat());
  const indices = new Uint16Array(triangles.flat());
  const indexOffset = vertices.byteLength;
  const binary = new ArrayBuffer(indexOffset + indices.byteLength);
  new Float32Array(binary, 0, vertices.length).set(vertices);
  new Uint16Array(binary, indexOffset, indices.length).set(indices);
  return packGlb({
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0, 1, 2] }],
    nodes: [
      { mesh: 0 },
      { mesh: 0, translation: [5, 0, 0], scale: [Math.sqrt(2.98), 0.1, 0.1] },
      { mesh: 0, translation: [8, 0, 0], scale: [0.1, 0.1, 0.1] },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: vertices.length / 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: vertices.byteLength },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength },
    ],
    buffers: [{ byteLength: binary.byteLength }],
  }, binary);
}
