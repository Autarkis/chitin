#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const sourcePath = path.join(repoRoot, "docs", "compilation-report.schema.json");
const typeScriptPath = path.join(
  repoRoot,
  "integrations",
  "wasm-lite",
  "src",
  "report-schema-constants.ts",
);

function loadSchema() {
  return JSON.parse(readFileSync(sourcePath, "utf8"));
}

const REQUIRED_CONSTANTS = [
  { name: "TOP_LEVEL_REQUIRED", get: (schema) => schema.required },
  { name: "VERDICT_REQUIRED", get: (schema) => schema.properties.verdict.required },
  {
    name: "CHECK_REQUIRED",
    get: (schema) => schema.properties.verdict.properties.checks.items.required,
  },
  { name: "INPUT_REQUIRED", get: (schema) => schema.properties.input.required },
  { name: "OUTPUT_REQUIRED", get: (schema) => schema.properties.output.required },
  { name: "WARNING_REQUIRED", get: (schema) => schema.$defs.warning.required },
  { name: "METRIC_REQUIRED", get: (schema) => schema.$defs.metric.required },
  { name: "PROCESSING_REQUIRED", get: (schema) => schema.properties.processing.required },
  {
    name: "FALLBACKS_REQUIRED",
    get: (schema) => schema.properties.processing.properties.fallbacks.required,
  },
  {
    name: "REFINEMENTS_REQUIRED",
    get: (schema) => schema.properties.processing.properties.refinements.required,
  },
  {
    name: "SNUG_FIT_REQUIRED",
    get: (schema) =>
      schema.properties.processing.properties.refinements.properties.snug_fit.required,
  },
  { name: "RUNTIME_REQUIRED", get: (schema) => schema.properties.runtime.required },
  {
    name: "REPRODUCIBILITY_REQUIRED",
    get: (schema) => schema.properties.reproducibility.required,
  },
  { name: "CONFIG_REQUIRED", get: (schema) => schema.properties.config.required },
];

const ENUM_CONSTANTS = [
  { name: "REPORT_STATUS_VALUES", get: (schema) => schema.properties.status.enum },
  {
    name: "VERDICT_STATUS_VALUES",
    get: (schema) => schema.properties.verdict.properties.status.enum,
  },
  {
    name: "CHECK_STATUS_VALUES",
    get: (schema) => schema.properties.verdict.properties.checks.items.properties.status.enum,
  },
  { name: "WARNING_SEVERITY_VALUES", get: (schema) => schema.$defs.warning.properties.severity.enum },
  { name: "METRIC_STATUS_VALUES", get: (schema) => schema.$defs.metric.properties.status.enum },
  {
    name: "SNUG_FIT_STATUS_VALUES",
    get: (schema) =>
      schema.properties.processing.properties.refinements.properties.snug_fit.properties.status
        .enum,
  },
];

function formatArray(values) {
  const items = values.map((value) => JSON.stringify(value));
  const oneLine = `[${items.join(", ")}]`;
  if (oneLine.length <= 80) return oneLine;
  return `[\n  ${items.join(",\n  ")},\n]`;
}

function renderConstants(schema, entries) {
  return entries
    .map(({ name, get }) => {
      const values = get(schema);
      if (!Array.isArray(values) || values.length === 0) {
        throw new Error(`${sourcePath} produced no values for ${name}`);
      }
      return `export const ${name} = ${formatArray(values)} as const;`;
    })
    .join("\n");
}

function generateTypeScript(schema) {
  return `// Generated from docs/compilation-report.schema.json; do not edit.

${renderConstants(schema, REQUIRED_CONSTANTS)}

${renderConstants(schema, ENUM_CONSTANTS)}
`;
}

const schema = loadSchema();
writeFileSync(typeScriptPath, generateTypeScript(schema), "utf8");
console.log(`wrote ${path.relative(repoRoot, typeScriptPath)}`);
