import type { DecomposeConfig } from "./types.js";
import {
  COACD_CONCAVITY_THRESHOLD,
  COACD_PREPROCESS_RESOLUTION,
} from "./shared-constants.js";

export const DECOMPOSE_DEFAULTS = Object.freeze({
  threshold: COACD_CONCAVITY_THRESHOLD,
  maxConvexHull: -1,
  prepResolution: COACD_PREPROCESS_RESOLUTION,
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
