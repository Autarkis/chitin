#!/usr/bin/env node
// Regenerates the typed metric-name constants shared by the Python and
// TypeScript compilation reports from the single source of truth,
// docs/metric-vocabulary.json. Run after editing that file:
//
//   node scripts/gen-metric-names.mjs
//
// Outputs:
//   src/chitin/_metric_names.py
//   integrations/wasm-lite/src/metric-names.ts

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

const vocabPath = path.join(repoRoot, "docs", "metric-vocabulary.json");
const pyOutPath = path.join(repoRoot, "src", "chitin", "_metric_names.py");
const tsOutPath = path.join(
  repoRoot,
  "integrations",
  "wasm-lite",
  "src",
  "metric-names.ts",
);

function toConstantName(metricName) {
  return metricName.toUpperCase();
}

function loadVocabulary() {
  const raw = readFileSync(vocabPath, "utf8");
  const doc = JSON.parse(raw);
  if (!doc.metrics || typeof doc.metrics !== "object") {
    throw new Error(`${vocabPath} is missing a top-level "metrics" object`);
  }
  return doc.metrics;
}

function collectEntries(metrics) {
  const categories = [];
  const seenNames = new Map();
  for (const [category, categoryMetrics] of Object.entries(metrics)) {
    const entries = [];
    for (const [name, spec] of Object.entries(categoryMetrics)) {
      if (seenNames.has(name)) {
        throw new Error(
          `duplicate metric name "${name}" in categories "${seenNames.get(name)}" and "${category}"`,
        );
      }
      seenNames.set(name, category);
      entries.push({
        name,
        constant: toConstantName(name),
        unit: spec.unit ?? null,
        description: spec.description ?? "",
        aliases: Array.isArray(spec.aliases) ? spec.aliases : [],
      });
    }
    categories.push({ category, entries });
  }
  return categories;
}

function pyStringLiteral(value) {
  return JSON.stringify(value);
}

function generatePython(categories) {
  const lines = [];
  lines.push('"""Canonical compilation-report metric names.');
  lines.push("");
  lines.push("Generated, do not edit by hand. Source of truth:");
  lines.push("docs/metric-vocabulary.json. Regenerate with:");
  lines.push("");
  lines.push("    node scripts/gen-metric-names.mjs");
  lines.push('"""');
  lines.push("");
  lines.push("from __future__ import annotations");
  lines.push("");

  for (const { category, entries } of categories) {
    lines.push(`# --- ${category} ---`);
    for (const entry of entries) {
      lines.push(`${entry.constant} = ${pyStringLiteral(entry.name)}`);
    }
    lines.push("");
  }

  lines.push("# Maps a legacy/alias metric name to its canonical replacement.");
  lines.push("ALIASES: dict[str, str] = {");
  for (const { entries } of categories) {
    for (const entry of entries) {
      for (const alias of entry.aliases) {
        lines.push(`    ${pyStringLiteral(alias)}: ${pyStringLiteral(entry.name)},`);
      }
    }
  }
  lines.push("}");
  lines.push("");

  return lines.join("\n");
}

function tsStringLiteral(value) {
  return JSON.stringify(value);
}

function generateTypeScript(categories) {
  const lines = [];
  lines.push("// Canonical compilation-report metric names.");
  lines.push("//");
  lines.push("// Generated, do not edit by hand. Source of truth:");
  lines.push("// docs/metric-vocabulary.json. Regenerate with:");
  lines.push("//");
  lines.push("//   node scripts/gen-metric-names.mjs");
  lines.push("");

  for (const { category, entries } of categories) {
    lines.push(`// --- ${category} ---`);
    for (const entry of entries) {
      lines.push(`export const ${entry.constant} = ${tsStringLiteral(entry.name)};`);
    }
    lines.push("");
  }

  lines.push("// Maps a legacy/alias metric name to its canonical replacement.");
  lines.push("export const ALIASES: Record<string, string> = {");
  for (const { entries } of categories) {
    for (const entry of entries) {
      for (const alias of entry.aliases) {
        lines.push(`  ${tsStringLiteral(alias)}: ${tsStringLiteral(entry.name)},`);
      }
    }
  }
  lines.push("};");
  lines.push("");

  return lines.join("\n");
}

function main() {
  const metrics = loadVocabulary();
  const categories = collectEntries(metrics);

  writeFileSync(pyOutPath, generatePython(categories), "utf8");
  writeFileSync(tsOutPath, generateTypeScript(categories), "utf8");

  console.log(`wrote ${path.relative(repoRoot, pyOutPath)}`);
  console.log(`wrote ${path.relative(repoRoot, tsOutPath)}`);
}

main();
