// Refuse to package without the built wasm artifacts. npm does not fail packing
// when files listed in "files" are missing, so check them explicitly.
import { statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const dir = join(dirname(fileURLToPath(import.meta.url)), "..");

const artifacts = [
  { file: "coacd.mjs", min: 40_000, max: 200_000 },
  { file: "coacd.wasm", min: 400_000, max: 800_000 },
  { file: "poisson.mjs", min: 20_000, max: 300_000 },
  { file: "poisson.wasm", min: 200_000, max: 2_000_000 },
];

let ok = true;
for (const { file, min, max } of artifacts) {
  let size;
  try {
    size = statSync(join(dir, file)).size;
  } catch {
    console.error(
      `prepack: missing ${file} — build it (integrations/wasm/build.sh) before packing`,
    );
    ok = false;
    continue;
  }
  if (size < min || size > max) {
    console.error(`prepack: ${file} size ${size} outside [${min}, ${max}]`);
    ok = false;
  } else {
    console.log(`prepack: ${file} ok (${size} bytes)`);
  }
}

if (!ok) process.exit(1);
