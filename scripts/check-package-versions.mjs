#!/usr/bin/env node
// Validates the npm-published package.json files agree on shared metadata
// (license, repository, homepage) and each carries the fields npm/consumers
// expect. Run from repo root: node scripts/check-package-versions.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

const PACKAGES = [
  "integrations/chitin-wasm/package.json",
  "integrations/wasm-lite/package.json",
  "integrations/web/package.json",
];

const REQUIRED_FIELDS = [
  ["name", (v) => typeof v === "string"],
  ["version", (v) => typeof v === "string" && /^\d+\.\d+\.\d+/.test(v)],
  ["license", (v) => typeof v === "string"],
  ["repository", (v) => typeof v === "object" && v !== null && typeof v.url === "string"],
  ["homepage", (v) => typeof v === "string"],
  ["description", (v) => typeof v === "string"],
  ["exports", (v) => typeof v === "object" && v !== null && !Array.isArray(v)],
  ["files", (v) => Array.isArray(v)],
];

const CONSISTENCY_FIELDS = [
  ["license", (pkg) => pkg.license],
  ["repository.url", (pkg) => pkg.repository?.url],
  ["homepage", (pkg) => pkg.homepage],
];

let failed = false;
const rows = [];

const pkgs = PACKAGES.map((rel) => {
  const abs = path.join(repoRoot, rel);
  const pkg = JSON.parse(readFileSync(abs, "utf8"));
  return { rel, pkg };
});

for (const { rel, pkg } of pkgs) {
  for (const [field, isValid] of REQUIRED_FIELDS) {
    if (!(field in pkg) || !isValid(pkg[field])) {
      console.error(`FAIL ${rel}: missing or invalid required field "${field}"`);
      failed = true;
    }
  }
  rows.push({
    name: pkg.name ?? "(missing)",
    version: pkg.version ?? "(missing)",
    license: pkg.license ?? "(missing)",
  });
}

for (const [label, getValue] of CONSISTENCY_FIELDS) {
  const values = new Map();
  for (const { rel, pkg } of pkgs) {
    const value = getValue(pkg);
    if (!values.has(value)) values.set(value, []);
    values.get(value).push(rel);
  }
  if (values.size > 1) {
    console.error(`FAIL ${label} differs across packages:`);
    for (const [value, files] of values) {
      console.error(`  ${JSON.stringify(value)}: ${files.join(", ")}`);
    }
    failed = true;
  }
}

const majors = new Map();
for (const { rel, pkg } of pkgs) {
  const major = String(pkg.version ?? "").split(".")[0];
  if (!majors.has(major)) majors.set(major, []);
  majors.get(major).push(rel);
}
if (majors.size > 1) {
  console.warn("WARN major versions differ across packages:");
  for (const [major, files] of majors) {
    console.warn(`  major ${major}: ${files.join(", ")}`);
  }
}

const nameWidth = Math.max(4, ...rows.map((r) => r.name.length));
const versionWidth = Math.max(7, ...rows.map((r) => r.version.length));
console.log();
console.log(`${"name".padEnd(nameWidth)}  ${"version".padEnd(versionWidth)}  license`);
for (const row of rows) {
  console.log(`${row.name.padEnd(nameWidth)}  ${row.version.padEnd(versionWidth)}  ${row.license}`);
}
console.log();

if (failed) {
  console.error("Package metadata consistency check FAILED");
  process.exit(1);
}
console.log("Package metadata consistency check passed");
process.exit(0);
