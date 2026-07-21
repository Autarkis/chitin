import RAPIER from "@dimforge/rapier3d-compat";
import { readFileSync } from "fs";
import { resolve } from "path";
import { beforeAll, describe, expect, it } from "vitest";

import { parsePhys, type PhysFile } from "../src/phys-parser.js";
import { addToWorld, applyBindPose, createColliders } from "../src/rapier.js";

const DIR = resolve(__dirname, "conformance");

function load(name: string): PhysFile {
  const b = readFileSync(resolve(DIR, name));
  return parsePhys(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

beforeAll(async () => {
  await RAPIER.init();
});

describe("rapier adapter", () => {
  it("createColliders builds one convex-mesh collider per hull", () => {
    const phys = load("static_hull.phys");
    const { colliders } = createColliders(RAPIER, phys);
    expect(colliders.length).toBe(phys.hulls.length);
    expect(colliders.length).toBeGreaterThan(0);
  });

  it("addToWorld attaches every hull's collider to a fixed body", () => {
    const phys = load("static_hull.phys");
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    const body = addToWorld(RAPIER, world, phys);
    expect(body.isFixed()).toBe(true);
    expect(body.numColliders()).toBe(phys.hulls.length);
  });

  it("addToWorld places a rigged .phys at its bind pose", () => {
    const phys = load("rigged.phys");
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    const body = addToWorld(RAPIER, world, phys);
    expect(body.isFixed()).toBe(true);
    // One collider per hull, all attached — no bone-local collapse, no throw.
    expect(body.numColliders()).toBe(phys.hulls.length);
  });

  it("addToWorld errors on bone-local hulls without bind poses", () => {
    // hasBones true but no bind poses: the hulls are bone-local yet there is no
    // transform to place them, so addToWorld must refuse rather than collapse.
    const phys = {
      ...load("static_hull.phys"),
      hasBindPoses: false,
      bones: [],
      hulls: load("static_hull.phys").hulls.map((h) => ({ ...h, boneIndex: 0 })),
    } as PhysFile;
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    expect(() => addToWorld(RAPIER, world, phys)).toThrow(/bind poses/);
  });

  it("applyBindPose reconstructs world = local @ bind_transform", () => {
    // Identity leaves bone-local vertices untouched.
    const local = new Float32Array([0.25, -0.5, 2]);
    const identity = new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]);
    expect([...applyBindPose(local, identity)]).toEqual([0.25, -0.5, 2]);

    // Row-major translation lives in the last row; a +1 X / +3 Y / -2 Z shift
    // must land on x/y/z respectively (row-vector convention, docs/phys.md).
    const translate = new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 3, -2, 1,
    ]);
    expect([...applyBindPose(local, translate)]).toEqual([1.25, 2.5, 0]);

    // Diagonal scale multiplies each axis (a channel a collider pose cannot hold).
    const scale = new Float32Array([
      2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1,
    ]);
    expect([...applyBindPose(local, scale)]).toEqual([0.5, -1.5, 8]);
  });

  it("addToWorld shifts a rigged hull by its bone's bind translation", () => {
    // rigged.phys: hull 1 -> bone "child" (row-major +1 X translation), so its
    // world-space vertices must all be offset +1 in X from the bone-local ones.
    const phys = load("rigged.phys");
    const hull1 = phys.hulls[1];
    expect(hull1.boneIndex).toBe(1);
    const world = applyBindPose(hull1.vertices, phys.bones[1].bindTransform);
    for (let i = 0; i < hull1.vertices.length; i += 3) {
      expect(world[i]).toBeCloseTo(hull1.vertices[i] + 1, 6);
      expect(world[i + 1]).toBeCloseTo(hull1.vertices[i + 1], 6);
      expect(world[i + 2]).toBeCloseTo(hull1.vertices[i + 2], 6);
    }
  });

  it("createColliders exposes a per-bone map for a rigged .phys", () => {
    const phys = load("rigged.phys");
    const { colliders, boneMap } = createColliders(RAPIER, phys);
    expect(colliders.length).toBe(phys.hulls.length);
    expect(boneMap.size).toBeGreaterThan(0);
  });
});
