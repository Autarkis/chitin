import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";
import { parsePhys } from "../src/phys-parser.js";

const FIXTURE = resolve(__dirname, "golden_rigged.phys");
const UNALIGNED_FIXTURE = resolve(__dirname, "unaligned_bind.phys");

function loadFixture(path = FIXTURE): ArrayBuffer {
  const buf = readFileSync(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function makeHeader(version = 3, flags = 0, extraBytes = 0): ArrayBuffer {
  const buf = new ArrayBuffer(32 + extraBytes);
  const view = new DataView(buf);
  view.setUint32(0, 0x53594850, true);
  view.setUint16(4, version, true);
  view.setUint16(6, flags, true);
  view.setUint32(20, 32, true);
  view.setUint32(24, 32, true);
  view.setUint32(28, 32, true);
  return buf;
}

describe("parsePhys", () => {
  it("reads header correctly", () => {
    const phys = parsePhys(loadFixture());
    expect(phys.version).toBe(2);
    expect(phys.hasBones).toBe(true);
    expect(phys.hasBindPoses).toBe(true);
    expect(phys.hasLod).toBe(false);
    expect(phys.hulls).toHaveLength(1);
    expect(phys.bones).toHaveLength(1);
  });

  it("reads bone name", () => {
    const phys = parsePhys(loadFixture());
    expect(phys.bones[0].name).toBe("test_bone");
  });

  it("reads hull-bone assignment", () => {
    const phys = parsePhys(loadFixture());
    expect(phys.hulls[0].boneIndex).toBe(0);
  });

  it("reads bind transform with translation at +5 X", () => {
    const phys = parsePhys(loadFixture());
    const bt = phys.bones[0].bindTransform;

    // Row-vector convention: translation in row 3 (indices 12..14)
    expect(bt[12]).toBeCloseTo(5.0, 4);
    expect(bt[13]).toBeCloseTo(0.0, 4);
    expect(bt[14]).toBeCloseTo(0.0, 4);

    // Rotation is identity (diagonal = 1)
    expect(bt[0]).toBeCloseTo(1.0, 4);
    expect(bt[5]).toBeCloseTo(1.0, 4);
    expect(bt[10]).toBeCloseTo(1.0, 4);
  });

  it("reconstructs world position from bone-local vertices", () => {
    const phys = parsePhys(loadFixture());
    const hull = phys.hulls[0];
    const bt = phys.bones[0].bindTransform;

    // Row-vector: world = local @ bindTransform
    // For identity rotation + translation [5,0,0]:
    // world_x = local_x * bt[0] + local_y * bt[4] + local_z * bt[8] + bt[12]
    let sumX = 0, sumY = 0, sumZ = 0;
    const n = hull.vertices.length / 3;
    for (let i = 0; i < n; i++) {
      const lx = hull.vertices[i * 3];
      const ly = hull.vertices[i * 3 + 1];
      const lz = hull.vertices[i * 3 + 2];
      sumX += lx * bt[0] + ly * bt[4] + lz * bt[8] + bt[12];
      sumY += lx * bt[1] + ly * bt[5] + lz * bt[9] + bt[13];
      sumZ += lx * bt[2] + ly * bt[6] + lz * bt[10] + bt[14];
    }

    expect(sumX / n).toBeCloseTo(5.0, 1);
    expect(sumY / n).toBeCloseTo(0.0, 1);
    expect(sumZ / n).toBeCloseTo(0.0, 1);
  });

  it("dequantizes vertices within AABB", () => {
    const phys = parsePhys(loadFixture());
    const hull = phys.hulls[0];
    for (let i = 0; i < hull.vertices.length / 3; i++) {
      for (let c = 0; c < 3; c++) {
        const v = hull.vertices[i * 3 + c];
        expect(v).toBeGreaterThanOrEqual(hull.aabbMin[c] - 0.01);
        expect(v).toBeLessThanOrEqual(hull.aabbMax[c] + 0.01);
      }
    }
  });

  it("parses unaligned bind-pose block without throwing", () => {
    const phys = parsePhys(loadFixture(UNALIGNED_FIXTURE));
    expect(phys.hasBones).toBe(true);
    expect(phys.hasBindPoses).toBe(true);
    expect(phys.bones).toHaveLength(1);
    expect(phys.bones[0].name).toBe("offset_bone");
    expect(phys.bones[0].bindTransform[12]).toBeCloseTo(3.0, 4);
  });

  it("rejects bad magic", () => {
    const bad = new ArrayBuffer(32);
    new Uint8Array(bad).set([0x4e, 0x4f, 0x50, 0x45]); // "NOPE"
    expect(() => parsePhys(bad)).toThrow("bad magic");
  });

  it("rejects unknown versions", () => {
    expect(() => parsePhys(makeHeader(999))).toThrow("unsupported .phys version");
  });

  it("rejects unknown flags", () => {
    expect(() => parsePhys(makeHeader(3, 0x8000))).toThrow("unknown flags");
  });

  it("rejects trailing bytes", () => {
    expect(() => parsePhys(makeHeader(3, 0, 4))).toThrow("trailing bytes");
  });

  it("skips LOD blocks structurally", () => {
    const buf = makeHeader(3, 0x04, 4);
    const view = new DataView(buf);
    view.setUint32(32, 0, true); // zero additional LOD tiers

    const phys = parsePhys(buf);
    expect(phys.hasLod).toBe(true);
    expect(phys.hulls).toHaveLength(0);
  });
});

describe("LOD tiers", () => {
  const MULTI_LOD = resolve(__dirname, "conformance", "multi_lod.phys");
  const NO_LOD = resolve(__dirname, "conformance", "static_hull.phys");

  it("parses tier concavities and hulls", async () => {
    const { parsePhys } = await import("../src/phys-parser.js");
    const phys = parsePhys(loadFixture(MULTI_LOD));
    expect(phys.hasLod).toBe(true);
    expect(phys.lodTiers.map((t) => Number(t.concavity.toFixed(2)))).toEqual([
      0.01, 0.05,
    ]);
    // tier 0 is the cube (8 verts), tier 1 the coarser tetra (4 verts)
    expect(phys.lodTiers[0].hulls[0].vertices.length / 3).toBe(8);
    expect(phys.lodTiers[1].hulls[0].vertices.length / 3).toBe(4);
  });

  it("selectLodHulls picks the nearest tier", async () => {
    const { parsePhys, selectLodHulls } = await import("../src/phys-parser.js");
    const phys = parsePhys(loadFixture(MULTI_LOD));
    expect(selectLodHulls(phys, 0.02)[0].vertices.length / 3).toBe(8); // -> 0.01
    expect(selectLodHulls(phys, 0.9)[0].vertices.length / 3).toBe(4); // -> 0.05
  });

  it("selectLodHulls falls back to LOD0 without tiers", async () => {
    const { parsePhys, selectLodHulls } = await import("../src/phys-parser.js");
    const phys = parsePhys(loadFixture(NO_LOD));
    expect(phys.lodTiers).toHaveLength(0);
    expect(selectLodHulls(phys, 0.3)).toBe(phys.hulls);
  });
});
