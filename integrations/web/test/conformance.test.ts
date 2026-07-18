import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";
import { parsePhys } from "../src/phys-parser.js";

// Cross-runtime .phys conformance — web reader half. Verifies parsePhys against the
// SAME frozen corpus + manifest the Python reader checks (tests/conformance, copied
// here so the web package is self-contained). Both readers parse LOD tier bodies, so
// the shared field set is version/flags/has* / hull count / per-hull AABB+counts /
// totals / bones / LOD tier concavity+count / validity.

const DIR = resolve(__dirname, "conformance");
const manifest: Record<string, any> = JSON.parse(
  readFileSync(resolve(DIR, "manifest.json"), "utf-8")
);

function load(name: string): ArrayBuffer {
  const buf = readFileSync(resolve(DIR, name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

const valid = Object.keys(manifest)
  .filter((k) => manifest[k].valid)
  .sort();
const invalid = Object.keys(manifest)
  .filter((k) => !manifest[k].valid)
  .sort();

describe("phys cross-runtime conformance", () => {
  it("has a corpus", () => {
    expect(valid.length).toBeGreaterThan(0);
    expect(invalid.length).toBeGreaterThan(0);
  });

  for (const name of valid) {
    it(`parses ${name} to manifest`, () => {
      const spec = manifest[name];
      const pf = parsePhys(load(name));

      expect(pf.version).toBe(spec.version);
      expect(pf.flags).toBe(spec.flags);
      expect(pf.hasBones).toBe(spec.hasBones);
      expect(pf.hasBindPoses).toBe(spec.hasBindPoses);
      expect(pf.hasLod).toBe(spec.hasLod);
      expect(pf.hulls).toHaveLength(spec.hullCount);

      // totals over the LOD0 hulls (what the web parser exposes)
      const totalV = pf.hulls.reduce((s, h) => s + h.vertices.length / 3, 0);
      const totalI = pf.hulls.reduce((s, h) => s + h.indices.length, 0);
      expect(totalV).toBe(spec.totalVertices);
      expect(totalI).toBe(spec.totalIndices);

      pf.hulls.forEach((h, i) => {
        const hs = spec.hulls[i];
        expect(h.vertices.length / 3).toBe(hs.vertexCount);
        expect(h.indices.length).toBe(hs.indexCount);
        expect(h.boneIndex).toBe(hs.boneIndex);
        for (let c = 0; c < 3; c++) {
          expect(h.aabbMin[c]).toBeCloseTo(hs.aabbMin[c], 4);
          expect(h.aabbMax[c]).toBeCloseTo(hs.aabbMax[c], 4);
        }
        if (h.indices.length > 0) {
          expect(Math.max(...h.indices)).toBeLessThan(h.vertices.length / 3);
        }
      });

      expect(pf.bones.map((b) => b.name)).toEqual(
        spec.bones.map((b: any) => b.name)
      );

      expect(pf.lodTiers).toHaveLength(spec.lodTiers.length);
      pf.lodTiers.forEach((t, i) => {
        expect(t.hulls).toHaveLength(spec.lodTiers[i].hullCount);
        expect(t.concavity).toBeCloseTo(spec.lodTiers[i].concavity, 4);
      });
    });
  }

  for (const name of invalid) {
    const spec = manifest[name];
    it(`rejects ${name}`, () => {
      // parsePhys throws, and the message names the defect (same needle Python checks)
      expect(() => parsePhys(load(name))).toThrow(spec.errorContains);
    });
  }
});
