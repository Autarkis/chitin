import type { DecomposeConfig } from "./types.js";

export const DECOMPOSE_DEFAULTS = Object.freeze({
  threshold: 0.05,
  maxConvexHull: -1,
  prepResolution: 50,
  sampleResolution: 2000,
  mctsNodes: 20,
  mctsIteration: 150,
  mctsMaxDepth: 3,
  maxChVertex: 256,
  merge: true,
}) satisfies Readonly<Required<DecomposeConfig>>;

export function resolveDecomposeConfig(
  config: DecomposeConfig = {},
): Required<DecomposeConfig> {
  const defined = Object.fromEntries(
    Object.entries(config).filter(([, v]) => v !== undefined),
  );
  return { ...DECOMPOSE_DEFAULTS, ...defined };
}
