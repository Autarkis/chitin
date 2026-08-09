import { readFileSync } from "fs";
import { resolve } from "path";
import { BufferAttribute, Group, Matrix4, Mesh, Vector3 } from "three";
import { describe, expect, it } from "vitest";

import { applyBindPose } from "../src/geometry.js";
import { parsePhys, type PhysFile } from "../src/phys-parser.js";
import {
  createDebugMeshes,
  createHullGeometries,
  createShadowProxy,
  physTransformFromObject3D,
} from "../src/three.js";

const DIR = resolve(__dirname, "conformance");

function load(name: string): PhysFile {
  const buffer = readFileSync(resolve(DIR, name));
  return parsePhys(
    buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength,
    ),
  );
}

describe("Three.js adapter", () => {
  it("builds bind-pose production geometry through the generic layer", () => {
    const phys = load("rigged.phys");
    const geometries = createHullGeometries(phys);
    const child = phys.hulls.findIndex((hull) => hull.boneIndex === 1);
    const positions = geometries[child].getAttribute(
      "position",
    ) as BufferAttribute;

    expect(positions.getX(0)).toBeCloseTo(phys.hulls[child].vertices[0] + 1, 6);
  });

  it("preserves the previous Three matrix result for a full affine bind", () => {
    const vertices = new Float32Array([1, 2, 3, -2, 1, 0.5]);
    const bind = new Float32Array([
      0, 2, 0, 0, -3, 0, 0.25, 0, 0.5, 0, 4, 0, 5, 6, 7, 1,
    ]);
    const matrix = new Matrix4().fromArray(bind);
    const materialized = applyBindPose(vertices, bind);

    for (let i = 0; i < vertices.length; i += 3) {
      const expected = new Vector3(
        vertices[i],
        vertices[i + 1],
        vertices[i + 2],
      ).applyMatrix4(matrix);
      expect(materialized[i]).toBeCloseTo(expected.x, 6);
      expect(materialized[i + 1]).toBeCloseTo(expected.y, 6);
      expect(materialized[i + 2]).toBeCloseTo(expected.z, 6);
    }
  });

  it("creates color-invisible casting meshes for each collision hull", () => {
    const phys = load("static_hull.phys");
    const proxy = createShadowProxy(phys);

    expect(proxy.children).toHaveLength(phys.hulls.length);
    for (const child of proxy.children) {
      const mesh = child as Mesh;
      expect(mesh.castShadow).toBe(true);
      expect(mesh.receiveShadow).toBe(false);
      expect(mesh.material).toMatchObject({
        colorWrite: false,
        depthWrite: false,
      });
    }
  });

  it("can merge hulls into one shadow draw object", () => {
    const phys = load("rigged.phys");
    const proxy = createShadowProxy(phys, { mode: "merged-hulls" });
    const mesh = proxy.children[0] as Mesh;
    const positionCount = phys.hulls.reduce(
      (total, hull) => total + hull.vertices.length / 3,
      0,
    );

    expect(proxy.children).toHaveLength(1);
    expect(mesh.geometry.getAttribute("position").count).toBe(positionCount);
  });

  it("debug geometry uses a requested collision LOD", () => {
    const phys = load("multi_lod.phys");
    const debug = createDebugMeshes(phys, {
      lodConcavity: 0.9,
      wireframe: false,
    });
    const mesh = debug.children[0] as Mesh;
    expect(mesh.geometry.getAttribute("position").count).toBe(4);
  });

  it("converts nested Three world transforms to the neutral TRS contract", () => {
    const parent = new Group();
    parent.position.set(10, 0, 0);
    const child = new Group();
    child.position.set(0, 2, 0);
    child.scale.set(2, 3, 4);
    parent.add(child);

    expect(physTransformFromObject3D(child)).toMatchObject({
      translation: [10, 2, 0],
      scale: [2, 3, 4],
    });
  });
});
