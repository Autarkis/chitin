// Functional gate for the CoACD WASM build. Loads the compiled module under
// Node, decomposes a known non-convex mesh, and asserts it splits into multiple
// convex hulls with real geometry. This exercises the actual decompose path --
// the one link the TypeScript-only `web` CI job cannot cover, since it never
// builds the wasm. Run after build.sh: `node test/decompose.test.cjs`.
const path = require("node:path");
const assert = require("node:assert");

const createCoACD = require(path.join(__dirname, "..", "dist", "coacd.js"));

// L-prism: an L-shaped polygon extruded to a slab. 12 vertices, 20 triangles,
// watertight. It is concave, so a correct decomposition yields >= 2 convex
// parts (empirically 2 at threshold 0.05).
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

// Marshal the Embind result into plain numbers, releasing every handle. The
// wasm heap grows on every call otherwise; mirrors chitin-lite's decompose().
function marshalHulls(result) {
  const hulls = [];
  const hullVec = result.hulls;
  try {
    for (let i = 0; i < hullVec.size(); i++) {
      const h = hullVec.get(i);
      const hv = h.vertices;
      const hi = h.indices;
      try {
        hulls.push({ vertexFloats: hv.size(), indexCount: hi.size() });
      } finally {
        hv.delete();
        hi.delete();
        h.delete?.();
      }
    }
  } finally {
    hullVec.delete();
    result.delete?.();
  }
  return hulls;
}

async function main() {
  const mod = await createCoACD();
  // Signature matches src/coacd_bind.cpp: verts, faces, threshold,
  // maxConvexHull, prepResolution, sampleResolution, mctsNodes, mctsIteration,
  // mctsMaxDepth, maxChVertex, merge.
  const result = mod.decompose(vertices, faces, 0.05, -1, 50, 2000, 20, 150, 3, 256, true);
  const hulls = marshalHulls(result);

  console.log(`decomposed L-prism into ${hulls.length} hull(s)`);
  assert.ok(hulls.length >= 2, `expected >= 2 hulls, got ${hulls.length}`);
  hulls.forEach((h, i) => {
    // A convex hull is at least a tetrahedron: >= 4 verts (12 floats), >= 4
    // triangles (12 indices), and the index count must be a multiple of 3.
    assert.ok(h.vertexFloats >= 12, `hull ${i}: too few vertex floats (${h.vertexFloats})`);
    assert.ok(
      h.indexCount >= 12 && h.indexCount % 3 === 0,
      `hull ${i}: bad index count (${h.indexCount})`,
    );
  });
  console.log("OK: CoACD WASM decompose functional test passed");
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
