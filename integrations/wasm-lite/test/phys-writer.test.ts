import { describe, it, expect } from "vitest";
import { quantizeHulls, writePhys } from "../src/phys-writer.js";
import { ChitinError } from "../src/errors.js";
import type { ConvexHull } from "../src/types.js";

function makeBoxHull(): ConvexHull {
  return {
    vertices: new Float32Array([
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
      1,
    ]),
    indices: new Uint32Array([
      0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7,
      3, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
    ]),
  };
}

describe("quantizeHulls", () => {
  it("computes correct AABB", () => {
    const [q] = quantizeHulls([makeBoxHull()]);
    expect(q.aabbMin).toEqual([0, 0, 0]);
    expect(q.aabbMax).toEqual([1, 1, 1]);
  });

  it("produces int16 vertices", () => {
    const [q] = quantizeHulls([makeBoxHull()]);
    expect(q.quantizedVertices).toBeInstanceOf(Int16Array);
    expect(q.quantizedVertices.length).toBe(24); // 8 verts * 3
  });

  it("quantized min maps to -32768, max maps to 32767", () => {
    const [q] = quantizeHulls([makeBoxHull()]);
    const minVal = Math.min(...Array.from(q.quantizedVertices));
    const maxVal = Math.max(...Array.from(q.quantizedVertices));
    expect(minVal).toBe(-32768);
    expect(maxVal).toBe(32767);
  });
});

describe("writePhys", () => {
  it("writes valid header", () => {
    const buf = writePhys([makeBoxHull()]);
    const view = new DataView(buf);

    expect(view.getUint32(0, true)).toBe(0x53594850); // PHYS magic
    expect(view.getUint16(4, true)).toBe(3); // version
    expect(view.getUint16(6, true)).toBe(0); // flags
    expect(view.getUint32(8, true)).toBe(1); // hull count
    expect(view.getUint32(12, true)).toBe(8); // total verts
    expect(view.getUint32(16, true)).toBe(36); // total indices
  });

  it("round-trips through dequantization", () => {
    const hull = makeBoxHull();
    const buf = writePhys([hull]);
    const view = new DataView(buf);

    const hullTableOff = view.getUint32(20, true);
    const vertexDataOff = view.getUint32(24, true);

    const aabbMin = [
      view.getFloat32(hullTableOff + 16, true),
      view.getFloat32(hullTableOff + 20, true),
      view.getFloat32(hullTableOff + 24, true),
    ];
    const aabbMax = [
      view.getFloat32(hullTableOff + 28, true),
      view.getFloat32(hullTableOff + 32, true),
      view.getFloat32(hullTableOff + 36, true),
    ];

    for (let v = 0; v < 8; v++) {
      for (let c = 0; c < 3; c++) {
        const q = view.getInt16(vertexDataOff + (v * 3 + c) * 2, true);
        const extent = aabbMax[c] - aabbMin[c];
        const e = extent === 0 ? 1.0 : extent;
        const dequantized = ((q + 32768) / 65535) * e + aabbMin[c];
        const original = hull.vertices[v * 3 + c];
        expect(Math.abs(dequantized - original)).toBeLessThan(0.001);
      }
    }
  });

  it("handles multiple hulls", () => {
    const hulls = [makeBoxHull(), makeBoxHull()];
    const buf = writePhys(hulls);
    const view = new DataView(buf);
    expect(view.getUint32(8, true)).toBe(2);
    expect(view.getUint32(12, true)).toBe(16); // 8 + 8 verts
  });
});

describe("writer input validation", () => {
  const good = (): ConvexHull => ({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]),
  });

  function expectInvalid(hull: ConvexHull, needle: string) {
    let err: unknown;
    try {
      writePhys([hull]);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ChitinError);
    expect((err as ChitinError).code).toBe("INVALID_MESH");
    expect((err as Error).message).toContain(needle);
  }

  it("accepts a well-formed hull", () => {
    expect(() => writePhys([good()])).not.toThrow();
  });

  it("rejects empty hulls", () => {
    expectInvalid(
      { vertices: new Float32Array(0), indices: new Uint32Array(0) },
      "empty",
    );
  });

  it("rejects non-triangle index arrays", () => {
    const h = good();
    h.indices = new Uint32Array([0, 1, 2, 0]);
    expectInvalid(h, "not triangles");
  });

  it("rejects vertex arrays not divisible by 3", () => {
    const h = good();
    h.vertices = new Float32Array([0, 0, 0, 1, 0]);
    expectInvalid(h, "multiple of 3");
  });

  it("rejects non-finite vertices", () => {
    const h = good();
    h.vertices[4] = Number.NaN;
    expectInvalid(h, "non-finite");
  });

  it("rejects indices past the vertex count", () => {
    const h = good();
    h.indices = new Uint32Array([0, 1, 99, 0, 1, 2]);
    expectInvalid(h, "out of range");
  });
});
