import { describe, expect, it } from "vitest";

import { evaluateColliderQuality } from "../src/quality.js";
import type { TriangleMesh } from "../src/mesh.js";
import type { ConvexHull } from "../src/types.js";

const CUBE_FACES = new Int32Array([
  0, 2, 1, 0, 3, 2,
  4, 5, 6, 4, 6, 7,
  0, 1, 5, 0, 5, 4,
  1, 2, 6, 1, 6, 5,
  2, 3, 7, 2, 7, 6,
  3, 0, 4, 3, 4, 7,
]);

function cubeVertices(min: number, max: number): number[] {
  return [
    min, min, min,
    max, min, min,
    max, max, min,
    min, max, min,
    min, min, max,
    max, min, max,
    max, max, max,
    min, max, max,
  ];
}

function cubeMesh(min = 0, max = 1): TriangleMesh {
  return {
    vertices: new Float64Array(cubeVertices(min, max)),
    faces: CUBE_FACES.slice(),
  };
}

function cubeHull(min = 0, max = 1): ConvexHull {
  return {
    vertices: new Float32Array(cubeVertices(min, max)),
    indices: new Uint32Array(CUBE_FACES),
  };
}

describe("evaluateColliderQuality", () => {
  it("reports complete surface coverage and no false fill for an exact collider", () => {
    const result = evaluateColliderQuality(cubeMesh(), [cubeHull()], {
      surfaceSamples: 256,
      volumeSamples: 512,
    });

    expect(result).toMatchObject({
      method: "deterministic_halton_v1",
      source_surface_coverage: 1,
      worst_component_surface_coverage: 1,
      collider_volume_precision: 1,
      false_fill_fraction: 0,
      deep_false_fill_fraction: 0,
      surface_samples: 256,
      volume_samples: 512,
      component_count: 1,
    });
  });

  it("detects collider volume that fills space outside the source", () => {
    const result = evaluateColliderQuality(cubeMesh(), [cubeHull(-1, 2)], {
      surfaceSamples: 256,
      volumeSamples: 4096,
    });

    expect(result.source_surface_coverage).toBe(1);
    expect(result.collider_volume_precision).not.toBeNull();
    expect(result.collider_volume_precision!).toBeLessThan(0.05);
    expect(result.false_fill_fraction!).toBeGreaterThan(0.95);
    expect(result.deep_false_fill_fraction!).toBeGreaterThan(0.9);
  });

  it("keeps disconnected-component coverage visible", () => {
    const first = cubeMesh();
    const second = cubeMesh(2, 3);
    const vertices = new Float64Array([...first.vertices, ...second.vertices]);
    const faces = new Int32Array([
      ...first.faces,
      ...Array.from(second.faces, (index) => index + first.vertices.length / 3),
    ]);
    const result = evaluateColliderQuality({ vertices, faces }, [cubeHull()], {
      surfaceSamples: 512,
      volumeSamples: 512,
    });

    expect(result.component_count).toBe(2);
    expect(result.source_surface_coverage).toBe(0.5);
    expect(result.worst_component_surface_coverage).toBe(0);
  });

  it("reports fit for each component when hull ownership is available", () => {
    const first = cubeMesh();
    const second = cubeMesh(2, 3);
    const firstHull = cubeHull();
    const secondHull = cubeHull(2, 3);
    const result = evaluateColliderQuality({
      vertices: new Float64Array([...first.vertices, ...second.vertices]),
      faces: new Int32Array([
        ...first.faces,
        ...Array.from(second.faces, (index) => index + first.vertices.length / 3),
      ]),
    }, [firstHull, secondHull], {
      surfaceSamples: 256,
      volumeSamples: 512,
      minColliderSamples: 8,
    }, [[firstHull], [secondHull]]);

    expect(result.components).toHaveLength(2);
    for (const component of result.components) {
      expect(component).toMatchObject({
        hull_count: 1,
        collider_triangle_count: 12,
        surface_coverage: 1,
        collider_volume_precision: 1,
        false_fill_fraction: 0,
        deep_false_fill_fraction: 0,
      });
    }
  });
});
