// Loads the compiled wasm under Node and asserts a concave mesh splits into
// multiple hulls. Run after build.sh: node test/decompose.test.mjs
import assert from "node:assert";

import createCoACD from "../dist/coacd.mjs";

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

const mod = await createCoACD();
// Signature matches src/coacd_bind.cpp: verts, faces, threshold, maxConvexHull,
// prepResolution, sampleResolution, mctsNodes, mctsIteration, mctsMaxDepth,
// maxChVertex, merge.
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

// Prove the binding actually enables CoACD's hull decimator. A triangulated
// sphere has far more convex-hull vertices than this cap; this failed when the
// binding passed decimate=false even though maxChVertex was exposed publicly.
function makeUvSphere(latitudeBands = 10, longitudeBands = 20) {
  const points = [[0, 1, 0]];
  for (let latitude = 1; latitude < latitudeBands; latitude++) {
    const phi = (Math.PI * latitude) / latitudeBands;
    for (let longitude = 0; longitude < longitudeBands; longitude++) {
      const theta = (2 * Math.PI * longitude) / longitudeBands;
      points.push([
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      ]);
    }
  }
  const bottom = points.length;
  points.push([0, -1, 0]);
  const triangles = [];
  for (let longitude = 0; longitude < longitudeBands; longitude++) {
    const next = (longitude + 1) % longitudeBands;
    triangles.push([0, 1 + next, 1 + longitude]);
  }
  for (let latitude = 0; latitude < latitudeBands - 2; latitude++) {
    const current = 1 + latitude * longitudeBands;
    const nextRing = current + longitudeBands;
    for (let longitude = 0; longitude < longitudeBands; longitude++) {
      const next = (longitude + 1) % longitudeBands;
      triangles.push(
        [current + longitude, current + next, nextRing + next],
        [current + longitude, nextRing + next, nextRing + longitude],
      );
    }
  }
  const lastRing = 1 + (latitudeBands - 2) * longitudeBands;
  for (let longitude = 0; longitude < longitudeBands; longitude++) {
    const next = (longitude + 1) % longitudeBands;
    triangles.push([lastRing + longitude, lastRing + next, bottom]);
  }
  return {
    vertices: new Float64Array(points.flat()),
    faces: new Int32Array(triangles.flat()),
  };
}

const sphere = makeUvSphere();
const decimatedResult = mod.decompose(
  sphere.vertices, sphere.faces, 1, 1, 50, 2000, 8, 40, 2, 12, true,
);
const decimatedHulls = marshalHulls(decimatedResult);
assert.equal(decimatedHulls.length, 1, "expected one convex sphere hull");
assert.ok(
  decimatedHulls[0].vertexFloats / 3 <= 12,
  `maxChVertex was not enforced (${decimatedHulls[0].vertexFloats / 3} vertices)`,
);
console.log("OK: CoACD WASM decompose functional test passed");
