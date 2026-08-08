import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { DECOMPOSE_DEFAULTS } from "../src/defaults.js";
import {
  BROWSER_PROFILE_NAMES,
  INTERACTIVE_MIN_HULL_VERTICES,
  NATIVE_MIN_HULL_VERTICES,
  PROFILE_NAMES,
} from "../src/shared-constants.js";

const CONTRACT = JSON.parse(
  readFileSync(new URL("../../../docs/shared-constants.json", import.meta.url), "utf8"),
);

describe("shared constants", () => {
  it("matches the cross-runtime contract", () => {
    expect(DECOMPOSE_DEFAULTS.threshold).toBe(CONTRACT.coacd.concavity_threshold);
    expect(DECOMPOSE_DEFAULTS.prepResolution).toBe(CONTRACT.coacd.preprocess_resolution);
    expect(NATIVE_MIN_HULL_VERTICES).toBe(CONTRACT.hull.native_min_vertices);
    expect(INTERACTIVE_MIN_HULL_VERTICES).toBe(CONTRACT.hull.interactive_min_vertices);
    expect(PROFILE_NAMES).toEqual(Object.keys(CONTRACT.profiles));
  });

  it("exposes exactly the explicitly implemented browser profiles", () => {
    expect(BROWSER_PROFILE_NAMES).toEqual(CONTRACT.browser_profiles);
  });
});
