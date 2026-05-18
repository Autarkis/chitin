import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";
import { parsePhys } from "../src/phys-parser.js";

const FIXTURE = resolve(__dirname, "golden_rigged.phys");

function loadFixture(): ArrayBuffer {
  const buf = readFileSync(FIXTURE);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe("parsePhys", () => {
  it("reads header correctly", () => {
    const phys = parsePhys(loadFixture());
    expect(phys.version).toBe(2);
    expect(phys.hasBones).toBe(true);
    expect(phys.hasBindPoses).toBe(true);
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

  it("rejects bad magic", () => {
    const bad = new ArrayBuffer(32);
    new Uint8Array(bad).set([0x4e, 0x4f, 0x50, 0x45]); // "NOPE"
    expect(() => parsePhys(bad)).toThrow("bad magic");
  });
});
