import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { CHITIN_LITE_VERSION } from "../src/version.js";

describe("package version", () => {
  it("keeps the report runtime version synchronized with package.json", () => {
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { version: string };
    expect(CHITIN_LITE_VERSION).toBe(packageJson.version);
  });
});
