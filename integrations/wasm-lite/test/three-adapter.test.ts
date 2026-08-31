import { describe, it, expect } from "vitest";
import type { BufferGeometry, Object3D, Mesh as ThreeMesh } from "three";
import { geometryToMesh, collectMeshes } from "../src/three-adapter.js";
import { ChitinError } from "../src/errors.js";

function mockAttribute(data: number[], itemSize: number) {
  const count = data.length / itemSize;
  return {
    count,
    getX(i: number) { return data[i * itemSize]; },
    getY(i: number) { return data[i * itemSize + 1]; },
    getZ(i: number) { return data[i * itemSize + 2]; },
  };
}

function mockIndex(data: number[]) {
  return {
    count: data.length,
    getX(i: number) { return data[i]; },
  };
}

function mockGeometry(options: {
  positions: number[];
  indices?: number[];
}): BufferGeometry {
  const position = mockAttribute(options.positions, 3);
  const index = options.indices ? mockIndex(options.indices) : null;
  return {
    getAttribute(name: string) { return name === "position" ? position : null; },
    getIndex() { return index; },
  } as unknown as BufferGeometry;
}

function identityMatrix(): number[] {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function translationMatrix(x: number, y: number, z: number): number[] {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

function mockMesh(geometry: BufferGeometry, matrix?: number[]): Object3D {
  return {
    isMesh: true,
    geometry,
    matrixWorld: { elements: matrix ?? identityMatrix() },
  } as unknown as Object3D;
}

function mockRoot(children: Object3D[]): Object3D {
  return {
    updateMatrixWorld() {},
    traverse(callback: (child: Object3D) => void) {
      callback(this as unknown as Object3D);
      for (const child of children) {
        callback(child);
      }
    },
  } as unknown as Object3D;
}

describe("geometryToMesh", () => {
  it("extracts indexed geometry", () => {
    const geometry = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
      indices: [0, 1, 2, 0, 2, 3],
    });
    const result = geometryToMesh(geometry);
    expect(result.vertices).toBeInstanceOf(Float64Array);
    expect(result.faces).toBeInstanceOf(Int32Array);
    expect(result.vertices.length).toBe(12); // 4 vertices * 3
    expect(result.faces.length).toBe(6); // 2 triangles * 3
    expect(Array.from(result.faces)).toEqual([0, 1, 2, 0, 2, 3]);
  });

  it("extracts non-indexed geometry", () => {
    const geometry = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
    });
    const result = geometryToMesh(geometry);
    expect(result.vertices.length).toBe(9); // 3 vertices * 3
    expect(result.faces.length).toBe(3); // 1 triangle * 3
    expect(Array.from(result.faces)).toEqual([0, 1, 2]);
  });

  it("throws INVALID_MESH for missing position attribute", () => {
    const geometry = {
      getAttribute() { return null; },
      getIndex() { return null; },
    } as unknown as BufferGeometry;
    expect(() => geometryToMesh(geometry)).toThrow(ChitinError);
    try {
      geometryToMesh(geometry);
    } catch (err) {
      expect((err as ChitinError).code).toBe("INVALID_MESH");
    }
  });

  it("throws INVALID_MESH for non-triangle index count", () => {
    const geometry = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1], // not a multiple of 3
    });
    expect(() => geometryToMesh(geometry)).toThrow(ChitinError);
  });
});

describe("collectMeshes", () => {
  it("merges multiple meshes", () => {
    const geom1 = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    });
    const geom2 = mockGeometry({
      positions: [2, 0, 0, 3, 0, 0, 2, 1, 0],
      indices: [0, 1, 2],
    });
    const root = mockRoot([
      mockMesh(geom1),
      mockMesh(geom2),
    ]);
    const result = collectMeshes(root);
    expect(result.vertices.length).toBe(18); // 6 vertices * 3
    expect(result.faces.length).toBe(6); // 2 triangles * 3
    // Second mesh faces should be offset by 3 (first mesh vertex count)
    expect(result.faces[3]).toBe(3);
    expect(result.faces[4]).toBe(4);
    expect(result.faces[5]).toBe(5);
  });

  it("applies world transform", () => {
    const geom = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    });
    const root = mockRoot([
      mockMesh(geom, translationMatrix(10, 20, 30)),
    ]);
    const result = collectMeshes(root);
    // First vertex should be translated
    expect(result.vertices[0]).toBe(10);
    expect(result.vertices[1]).toBe(20);
    expect(result.vertices[2]).toBe(30);
  });

  it("throws INVALID_MESH when no meshes found", () => {
    const root = mockRoot([]);
    expect(() => collectMeshes(root)).toThrow(ChitinError);
    try {
      collectMeshes(root);
    } catch (err) {
      expect((err as ChitinError).code).toBe("INVALID_MESH");
    }
  });

  it("skips non-mesh children", () => {
    const geom = mockGeometry({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    });
    const nonMesh = { isMesh: false } as unknown as Object3D;
    const root = mockRoot([nonMesh, mockMesh(geom)]);
    const result = collectMeshes(root);
    expect(result.vertices.length).toBe(9); // only 1 mesh's vertices
  });
});
