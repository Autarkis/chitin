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
  const profiles = Object.keys(contract.profiles ?? {}).filter((name) => name !== "$comment");
  if (profiles.length === 0) {
    throw new Error(`${sourcePath} contains no profiles`);
  }
  return { contract, profiles };
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
`;
}

function generateTypeScript(contract, profiles) {
  const profileValues = profiles.map((name) => JSON.stringify(name)).join(", ");
  return `// Generated from docs/shared-constants.json; do not edit.

export const COACD_CONCAVITY_THRESHOLD = ${JSON.stringify(contract.coacd.concavity_threshold)};
export const COACD_PREPROCESS_RESOLUTION = ${JSON.stringify(contract.coacd.preprocess_resolution)};
export const NATIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.native_min_vertices)};
export const INTERACTIVE_MIN_HULL_VERTICES = ${JSON.stringify(contract.hull.interactive_min_vertices)};
export const PROFILE_NAMES = [${profileValues}] as const;
export type ProfileName = (typeof PROFILE_NAMES)[number];
`;
}

const { contract, profiles } = loadContract();
writeFileSync(pythonPath, generatePython(contract, profiles), "utf8");
writeFileSync(typeScriptPath, generateTypeScript(contract, profiles), "utf8");
console.log(`wrote ${path.relative(repoRoot, pythonPath)}`);
console.log(`wrote ${path.relative(repoRoot, typeScriptPath)}`);
