export interface DecomposeConfig {
  threshold?: number;
  maxConvexHull?: number;
  prepResolution?: number;
  sampleResolution?: number;
  mctsNodes?: number;
  mctsIteration?: number;
  mctsMaxDepth?: number;
  maxChVertex?: number;
  merge?: boolean;
}

export interface ConvexHull {
  vertices: Float32Array; // flat xyz, length N*3
  indices: Uint32Array; // flat triangle indices, length M*3
}

export interface DecomposeResult {
  hulls: ConvexHull[];
}

export interface QuantizedHull {
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  quantizedVertices: Int16Array;
  indices: Uint16Array;
}
