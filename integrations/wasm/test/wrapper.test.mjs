// Release smoke test: lite -> CoACD -> .phys -> web.
// Requires wasm-lite and web to be built. Run from the repo root:
//   node integrations/wasm/test/wrapper.test.mjs
import assert from "node:assert";

import createCoACD from "../dist/coacd.mjs";
import { setModuleFactory, decompose, writePhys } from "../../wasm-lite/dist/index.js";
import { parsePhys, selectLodHulls } from "../../web/dist/index.js";

setModuleFactory(createCoACD);

// L-prism: concave, so a correct decomposition yields >= 2 convex hulls.
// prettier-ignore
const vertices = new Float64Array([
  -0.48, -0.58, -0.25,  0.72, -0.58, -0.25,  0.72,  0.02, -0.25,
   0.12,  0.02, -0.25,  0.12,  0.82, -0.25, -0.48,  0.82, -0.25,
  -0.48, -0.58,  0.25,  0.72, -0.58,  0.25,  0.72,  0.02,  0.25,
   0.12,  0.02,  0.25,  0.12,  0.82,  0.25, -0.48,  0.82,  0.25,
]);
// prettier-ignore
const faces = new Int32Array([
  2, 1, 0,   5, 4, 3,   3, 2, 0,   0, 5, 3,   6, 7, 8,
  9, 10, 11, 6, 8, 9,   9, 11, 6,  7, 6, 1,   1, 6, 0,
  8, 7, 2,   2, 7, 1,   9, 8, 3,   3, 8, 2,   10, 9, 4,
  4, 9, 3,   6, 11, 0,  0, 11, 5,  11, 10, 5, 5, 10, 4,
]);

const result = await decompose(vertices, faces, { threshold: 0.05 });
assert.ok(result.hulls.length >= 2, `expected >= 2 hulls, got ${result.hulls.length}`);

const buf = writePhys(result.hulls); // chitin-lite writer
const phys = parsePhys(buf); // chitin-web reader
assert.strictEqual(phys.version, 3, `expected .phys version 3, got ${phys.version}`);

const hulls = selectLodHulls(phys, 0);
assert.strictEqual(
  hulls.length,
  result.hulls.length,
  `reparsed hull count ${hulls.length} != decomposed ${result.hulls.length}`,
);

console.log(
  `OK: lite -> wasm -> writePhys -> web (${hulls.length} hulls, .phys v${phys.version}, ${buf.byteLength} bytes)`,
);
