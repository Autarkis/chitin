import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parseGlb } from "@autarkis/chitin-lite";
import { packGlb } from "../../../scripts/glb-pack.mjs";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(root, "public/assets");

const samples = [
  {
    output: "clearcoat-wicker.glb",
    source: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/ClearcoatWicker/glTF-Binary/ClearcoatWicker.glb",
    sha256: "f162b0cd7f8e6b7cef211eec57762165a78039676b8592ce1f965e2ddb34e843",
  },
  {
    output: "iridescent-dish-with-olives.glb",
    source: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/IridescentDishWithOlives/glTF-Binary/IridescentDishWithOlives.glb",
    sha256: "1540b4a36b790a907f4824cfe848ba481b3da3cc8070172b7b3ba178f78a7ed1",
  },
  {
    output: "barramundi-fish.glb",
    source: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/BarramundiFish/glTF-Binary/BarramundiFish.glb",
    sha256: "ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a",
  },
];

function hash(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function flattenedGeometryGlb(source) {
  const sourceBuffer = source.buffer.slice(
    source.byteOffset,
    source.byteOffset + source.byteLength,
  );
  const mesh = parseGlb(sourceBuffer);
  const positions = Float32Array.from(mesh.vertices);
  const indices = Uint32Array.from(mesh.faces);
  const indexOffset = (positions.byteLength + 3) & ~3;
  const binary = new Uint8Array(indexOffset + indices.byteLength);
  binary.set(new Uint8Array(positions.buffer));
  binary.set(new Uint8Array(indices.buffer), indexOffset);

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], positions[i + axis]);
      max[axis] = Math.max(max[axis], positions[i + axis]);
    }
  }

  return packGlb({
    asset: { version: "2.0", generator: "Chitin geometry-only sample preparation" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: "VEC3", min, max },
      { bufferView: 1, componentType: 5125, count: indices.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength, target: 34963 },
    ],
    buffers: [{ byteLength: binary.byteLength }],
  }, binary);
}

await mkdir(outputDirectory, { recursive: true });
for (const sample of samples) {
  const response = await fetch(sample.source);
  if (!response.ok) throw new Error(`${sample.source} returned HTTP ${response.status}`);
  const source = new Uint8Array(await response.arrayBuffer());
  const sourceHash = hash(source);
  if (sourceHash !== sample.sha256) {
    throw new Error(`${sample.output} source hash changed: expected ${sample.sha256}, got ${sourceHash}`);
  }
  const output = flattenedGeometryGlb(source);
  await writeFile(resolve(outputDirectory, sample.output), output);
  console.log(`${sample.output}\t${output.byteLength}\t${hash(output)}`);
}
