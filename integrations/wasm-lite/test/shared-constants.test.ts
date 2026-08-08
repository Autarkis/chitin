import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { DECOMPOSE_DEFAULTS } from "../src/defaults.js";
import {
  INTERACTIVE_MIN_HULL_VERTICES,
  NATIVE_MIN_HULL_VERTICES,
  PROFILE_NAMES,
} from "../src/shared-constants.js";

describe("shared constants", () => {
  it("matches the cross-runtime contract", () => {
    const contract = JSON.parse(
      readFileSync(new URL("../../../docs/shared-constants.json", import.meta.url), "utf8"),
    );

    expect(DECOMPOSE_DEFAULTS.threshold).toBe(contract.coacd.concavity_threshold);
    expect(DECOMPOSE_DEFAULTS.prepResolution).toBe(contract.coacd.preprocess_resolution);
    expect(NATIVE_MIN_HULL_VERTICES).toBe(contract.hull.native_min_vertices);
    expect(INTERACTIVE_MIN_HULL_VERTICES).toBe(contract.hull.interactive_min_vertices);
    expect(PROFILE_NAMES).toEqual(
      Object.keys(contract.profiles).filter((name) => name !== "$comment"),
    );
  });
});
