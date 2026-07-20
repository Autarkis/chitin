import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";

import { parsePhys } from "../src/phys-parser.js";

const DIR = resolve(__dirname, "conformance");

function load(name: string): ArrayBuffer {
  const buf = readFileSync(resolve(DIR, name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

// v3 header offsets (little-endian): hull_table_off @20, index_data_off @28.
const HULL_TABLE_OFF = 20;
const INDEX_DATA_OFF = 28;

describe("parsePhys structural validation", () => {
  it("accepts the valid fixtures unchanged", () => {
    expect(() => parsePhys(load("static_hull.phys"))).not.toThrow();
    expect(() => parsePhys(load("multi_lod.phys"))).not.toThrow();
    expect(() => parsePhys(load("rigged.phys"))).not.toThrow();
  });

  it("rejects a triangle index past the hull's vertex count", () => {
    const buf = load("static_hull.phys");
    const view = new DataView(buf);
    const indexDataOff = view.getUint32(INDEX_DATA_OFF, true);
    view.setUint16(indexDataOff, 0xffff, true); // first index -> wildly out of range
    expect(() => parsePhys(buf)).toThrow(/index .* >= hull vertex count/);
  });

  // In a hull descriptor, aabb_min[0] is at +16 (after vOff/vCount/iOff/iCount).
  const AABB_MIN0 = 16;

  it("rejects an inverted AABB (min > max)", () => {
    const buf = load("static_hull.phys");
    const view = new DataView(buf);
    const hullTableOff = view.getUint32(HULL_TABLE_OFF, true);
    view.setFloat32(hullTableOff + AABB_MIN0, 1e6, true); // min[0] huge -> exceeds max[0]
    expect(() => parsePhys(buf)).toThrow(/aabb_min.*>.*aabb_max/);
  });

  it("rejects a non-finite AABB coordinate", () => {
    const buf = load("static_hull.phys");
    const view = new DataView(buf);
    const hullTableOff = view.getUint32(HULL_TABLE_OFF, true);
    view.setFloat32(hullTableOff + AABB_MIN0, NaN, true);
    expect(() => parsePhys(buf)).toThrow(/non-finite aabb/);
  });

  it("rejects unreferenced index payload (hull sums < declared total)", () => {
    // Shrink the single hull's index count so it no longer accounts for every
    // declared index -- leaving hidden bytes the descriptors don't reference.
    const buf = load("static_hull.phys");
    const view = new DataView(buf);
    const hullTableOff = view.getUint32(HULL_TABLE_OFF, true);
    const iCount = view.getUint32(hullTableOff + 12, true); // descriptor: iCount @ +12
    view.setUint32(hullTableOff + 12, iCount - 3, true); // drop one triangle
    expect(() => parsePhys(buf)).toThrow(/indices sum to .* but total_indices/);
  });
});
