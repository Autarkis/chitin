import { describe, expect, it } from "vitest";

import { reconstructGaussianFieldMesh } from "../src/gaussian-field.js";
import { analyzeManifold } from "../src/manifold.js";
import { compilerWith, CompletingWorker } from "./compiler-fixture.js";

const FIELD = {
  centers: new Float32Array([0, 0, 0, 0.45, 0, 0]),
  scales: new Float32Array([0.5, 0.4, 0.35, 0.3, 0.28, 0.25]),
  quaternions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1]),
  opacities: new Float32Array([1, 0.9]),
};

describe("Gaussian field ingestion", () => {
  it("reconstructs a closed manifold voxel surface", () => {
    const mesh = reconstructGaussianFieldMesh(FIELD, { resolution: 28 });
    expect(mesh.vertices.length).toBeGreaterThan(0);
    expect(mesh.faces.length).toBeGreaterThan(0);
    expect(analyzeManifold(mesh.vertices, mesh.faces)).toMatchObject({
      manifold: true,
      boundary_edge_count: 0,
      non_manifold_edge_count: 0,
      inconsistent_winding_edge_count: 0,
    });
  });

  it("compiles reconstructed support through Chitin into .phys", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGaussianField(FIELD, {
      reconstruction: { resolution: 24 },
    });
    expect(result.phys.byteLength).toBeGreaterThan(0);
    expect(result.hulls).toHaveLength(1);
    expect(result.source).toMatchObject({ mesh_count: 1, primitive_count: 1 });
    expect(result.source.triangle_count).toBeGreaterThan(0);
    compiler.terminate();
  });

  it("rejects malformed fields before invoking decomposition", () => {
    expect(() =>
      reconstructGaussianFieldMesh({
        centers: new Float32Array([0, 0, 0]),
        scales: new Float32Array([1, 1]),
        quaternions: new Float32Array([0, 0, 0, 1]),
      }),
    ).toThrow(/scales must contain 3 values/);
  });
});
