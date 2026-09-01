#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const sourcePath = path.join(repoRoot, "docs", "shared-constants.json");
const pythonPath = path.join(repoRoot, "src", "chitin", "_shared_constants.py");
const typeScriptPath = path.join(
  repoRoot,
  "integrations",
  "wasm-lite",
  "src",
  "shared-constants.ts",
);

function loadContract() {
  const contract = JSON.parse(readFileSync(sourcePath, "utf8"));
  const numbers = [
    contract.coacd?.concavity_threshold,
    contract.coacd?.preprocess_resolution,
    contract.hull?.native_min_vertices,
    contract.hull?.interactive_min_vertices,
  ];
  if (numbers.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`${sourcePath} contains an invalid numeric constant`);
  }
  const profiles = Object.keys(contract.profiles ?? {});
  if (profiles.length === 0) {
    throw new Error(`${sourcePath} contains no profiles`);
  }
  const browserProfiles = contract.browser_profiles;
  if (!Array.isArray(browserProfiles) || browserProfiles.length === 0) {
    throw new Error(`${sourcePath} contains no browser profiles`);
  }
  const unknownBrowserProfiles = browserProfiles.filter((name) => !profiles.includes(name));
  if (unknownBrowserProfiles.length > 0) {
    throw new Error(
      `${sourcePath} lists unknown browser profiles: ${unknownBrowserProfiles.join(", ")}`,
    );
  }
  const thresholds = contract.acceptance_thresholds;
  if (!thresholds || typeof thresholds !== "object" || Array.isArray(thresholds)) {
    throw new Error(`${sourcePath} contains no acceptance thresholds`);
  }
  for (const [profile, values] of Object.entries(thresholds)) {
    if (!profiles.includes(profile) || !values || typeof values !== "object" || Array.isArray(values)) {
      throw new Error(`${sourcePath} contains invalid thresholds for ${profile}`);
    }
    if (Object.values(values).some((value) => typeof value !== "number" || !Number.isFinite(value))) {
      throw new Error(`${sourcePath} contains a non-numeric threshold for ${profile}`);
    }
  }
  return { contract, profiles, browserProfiles };
}

function generatePython(contract, profiles) {
  const profileValues = profiles.map((name) => JSON.stringify(name)).join(", ");
  const tupleSuffix = profiles.length === 1 ? "," : "";
  return `"""Generated from docs/shared-constants.json; do not edit."""

COACD_CONCAVITY_THRESHOLD = ${JSON.stringify(contract.coacd.concavity_threshold)}
COACD_PREPROCESS_RESOLUTION = ${JSON.stringify(contract.coacd.preprocess_resolution)}
NATIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.native_min_vertices)}
INTERACTIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.interactive_min_vertices)}
PROFILE_NAMES = (${profileValues}${tupleSuffix})
ACCEPTANCE_THRESHOLDS = ${JSON.stringify(contract.acceptance_thresholds, null, 2)}
`;
}

function generateTypeScript(contract, profiles, browserProfiles) {
  const profileValues = profiles.map((name) => JSON.stringify(name)).join(", ");
  const browserValues = browserProfiles.map((name) => JSON.stringify(name)).join(", ");
  return `// Generated from docs/shared-constants.json; do not edit.

export const COACD_CONCAVITY_THRESHOLD = ${JSON.stringify(contract.coacd.concavity_threshold)};
export const COACD_PREPROCESS_RESOLUTION = ${JSON.stringify(contract.coacd.preprocess_resolution)};
export const NATIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.native_min_vertices)};
export const INTERACTIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.interactive_min_vertices)};
export const PROFILE_NAMES = [${profileValues}] as const;
export type ProfileName = (typeof PROFILE_NAMES)[number];
export const BROWSER_PROFILE_NAMES = [${browserValues}] as const;
export type BrowserProfileName = (typeof BROWSER_PROFILE_NAMES)[number];
export const ACCEPTANCE_THRESHOLDS = ${JSON.stringify(contract.acceptance_thresholds, null, 2)} as const;
`;
}

const { contract, profiles, browserProfiles } = loadContract();
writeFileSync(pythonPath, generatePython(contract, profiles), "utf8");
writeFileSync(typeScriptPath, generateTypeScript(contract, profiles, browserProfiles), "utf8");
console.log(`wrote ${path.relative(repoRoot, pythonPath)}`);
console.log(`wrote ${path.relative(repoRoot, typeScriptPath)}`);
