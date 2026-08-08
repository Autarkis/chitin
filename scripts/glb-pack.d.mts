export function packGlb(
  document: object,
  binary: ArrayBuffer | ArrayBufferView,
  declaredBinaryLength?: number,
): ArrayBuffer;

export interface BuildMinimalGltfOptions {
  generator?: string;
  mode?: number;
  bounds?: boolean;
  min?: number[];
  max?: number[];
}

export interface MinimalGltf {
  document: object;
  binary: Uint8Array;
}

export function buildMinimalGltf(
  positions: Float32Array,
  indices: Uint16Array | Uint32Array,
  options?: BuildMinimalGltfOptions,
): MinimalGltf;
