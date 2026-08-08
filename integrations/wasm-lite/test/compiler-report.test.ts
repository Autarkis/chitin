import { describe, expect, it } from "vitest";

import { CompletingWorker, compilerWith } from "./compiler-fixture.js";
import { makeGlb } from "./glb-fixture.js";

describe("ChitinCompiler report assembly", () => {
  it("snapshots report.config.effective for a representative compile", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), { decompose: { threshold: 0.05 } });
    compiler.terminate();

    expect(result.report.config.effective).toMatchInlineSnapshot(`
      {
        "adaptive_hull_vertices": true,
        "component_count": 2,
        "component_plans": [
          {
            "allocation_weight": 1.5,
            "component_index": 0,
            "diagonal_ratio": 0.12803687993289598,
            "hollow_shell": false,
            "importance": 1,
            "max_hull_vertices": 4,
            "max_hulls": 64,
            "occupancy_ratio": 1,
            "output_hulls": 1,
            "output_triangles": 1,
            "roundness": 0,
            "simplified": false,
            "threshold": 0.1,
            "triangle_count": 1,
            "volume_ratio": 1,
          },
          {
            "allocation_weight": 1.5,
            "component_index": 1,
            "diagonal_ratio": 0.12803687993289598,
            "hollow_shell": false,
            "importance": 1,
            "max_hull_vertices": 4,
            "max_hulls": 64,
            "occupancy_ratio": 1,
            "output_hulls": 1,
            "output_triangles": 1,
            "roundness": 0,
            "simplified": false,
            "threshold": 0.1,
            "triangle_count": 1,
            "volume_ratio": 1,
          },
        ],
        "detail_budget_coarse_ratio": 0.7,
        "detail_budget_coarse_threshold": 0.6,
        "detail_budget_fine_threshold": 0.1,
        "detail_budget_ratio": 1,
        "detailed_component_min_threshold": 0.1,
        "detailed_component_threshold": 0.1,
        "effective_component_hull_vertices_max": 4,
        "effective_component_hull_vertices_mean": 4,
        "effective_component_hull_vertices_min": 4,
        "effective_component_threshold_max": 0.1,
        "effective_component_threshold_min": 0.1,
        "guarded_hollow_shell_component_count": 0,
        "hollow_shell_component_count": 0,
        "hollow_shell_max_occupancy_ratio": 0.05,
        "hollow_shell_max_threshold": 0.05,
        "hollow_shell_min_hulls": 8,
        "hollow_shell_threshold": null,
        "hull_vertex_roundness_metric": "isoperimetric_quotient",
        "importance_guarded_component_count": 0,
        "important_component_max_occupancy_ratio": 0.5,
        "important_component_max_threshold": 0.14,
        "max_hull_vertices": 96,
        "max_hulls": 128,
        "max_hulls_ceiling": 128,
        "max_workers": 1,
        "mcts_iterations": 40,
        "mcts_max_depth": 2,
        "mcts_nodes": 8,
        "min_hull_vertices": 8,
        "requested_max_hull_vertices": 256,
        "simplified_component_count": 0,
        "small_component_max_diagonal_ratio": 0.2,
        "small_component_max_volume_ratio": 0.005,
        "small_component_threshold": 1,
      }
    `);
  });

  it("pins aggregate and per-component quality metrics in component order", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      checkManifold: false,
      quality: { surfaceSamples: 32, volumeSamples: 64, minColliderSamples: 1 },
    });
    compiler.terminate();

    const metrics = result.report.metrics;
    expect(metrics.quality_component_count).toEqual({ value: 2, unit: "count", status: "measured" });
    expect(metrics.quality_surface_samples).toEqual({ value: 32, unit: "count", status: "measured" });
    expect(metrics.quality_volume_samples).toEqual({ value: 64, unit: "count", status: "measured" });
    expect(metrics.quality_component_0_vertex_count).toEqual({ value: 3, unit: "count", status: "measured" });
    expect(metrics.quality_component_0_triangle_count).toEqual({ value: 1, unit: "count", status: "measured" });

    const componentKeys = Object.keys(metrics).filter((key) => /^quality_component_\d+_/.test(key));
    const orderedByIndex = [...componentKeys].sort((left, right) => {
      const leftIndex = Number(left.match(/^quality_component_(\d+)_/)?.[1] ?? -1);
      const rightIndex = Number(right.match(/^quality_component_(\d+)_/)?.[1] ?? -1);
      return leftIndex - rightIndex;
    });
    expect(componentKeys).toEqual(orderedByIndex);
    expect(componentKeys[0]).toMatch(/^quality_component_0_/);
  });
});
