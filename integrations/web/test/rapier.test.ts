import RAPIER from "@dimforge/rapier3d-compat";
import { readFileSync } from "fs";
import { resolve } from "path";
import { beforeAll, describe, expect, it } from "vitest";

import { parsePhys, type PhysFile } from "../src/phys-parser.js";
import { addToWorld, createColliders } from "../src/rapier.js";

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

  it("addToWorld rejects a rigged .phys (bone-local hulls)", () => {
    const phys = load("rigged.phys");
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    expect(() => addToWorld(RAPIER, world, phys)).toThrow(/rigged/);
  });

  it("createColliders exposes a per-bone map for a rigged .phys", () => {
    const phys = load("rigged.phys");
    const { colliders, boneMap } = createColliders(RAPIER, phys);
    expect(colliders.length).toBe(phys.hulls.length);
    expect(boneMap.size).toBeGreaterThan(0);
  });
});
