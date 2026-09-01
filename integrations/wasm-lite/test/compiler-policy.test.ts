import { describe, expect, it } from "vitest";

import {
  CompletingWorker,
  compilerWith,
  makeScaledTetrahedraGlb,
} from "./compiler-fixture.js";
import {
  makeAdaptiveHullBudgetGlb,
  makeGlb,
  makeThinOpenTrayGlb,
} from "./glb-fixture.js";

describe("ChitinCompiler component policy", () => {
  it("rejects unknown profiles instead of attaching an inert label", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(
      compiler.compileGlb(makeGlb(), { profile: "unknown" as any }),
    ).rejects.toMatchObject({ code: "INVALID_CONFIG", stage: "validating-input" });
    compiler.terminate();
  });

  it("requires enough component-policy budget for disconnected components", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(makeGlb(), {
      componentPolicy: { maxHulls: 1 },
    })).rejects.toMatchObject({
      code: "INVALID_CONFIG",
      stage: "validating-input",
      context: { component_count: 2, max_hulls: 1 },
    });
    compiler.terminate();
  });

  it("reserves component-policy capacity for every disconnected component", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    await compiler.compileGlb(makeGlb(), { componentPolicy: { maxHulls: 3 } });
    expect(worker.configs).toEqual([
      { mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, threshold: 0.1, maxChVertex: 4, maxConvexHull: 2 },
      { mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, threshold: 0.1, maxChVertex: 4, maxConvexHull: 1 },
    ]);
    compiler.terminate();
  });

  it("keeps decompose.maxConvexHull as a per-component low-level cap", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    await compiler.compileGlb(makeGlb(), { decompose: { maxConvexHull: 1 } });
    expect(worker.configs.map((config) => config.maxConvexHull)).toEqual([1, 1]);
    compiler.terminate();
  });

  it("checks every GLB component for browser-incompatible open geometry by default", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    await compiler.compileGlb(makeGlb());
    expect(worker.manifoldChecks).toEqual([true, true]);
    compiler.terminate();
  });

  it("uses a deterministic scene-aware budget and reuses small-part results across detail changes", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const file = new Blob([makeScaledTetrahedraGlb()]);
    const firstProgress: Array<{ completed?: number; total?: number }> = [];
    const first = await compiler.compileGlb(file, {
      decompose: { threshold: 0.05 },
      onProgress: (progress) => {
        if (progress.stage === "decomposing") firstProgress.push(progress);
      },
    });
    expect(worker.configs).toEqual([
      { threshold: 0.1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 127 },
      { threshold: 1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 1 },
    ]);
    expect(firstProgress.map(({ completed, total }) => [completed, total])).toContainEqual([2, 2]);
    expect(first.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED",
    }));
    expect(first.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_THRESHOLD_CLAMPED",
    }));
    expect(first.report.config.effective).toMatchObject({
      max_hulls: 128,
      max_hulls_ceiling: 128,
      detail_budget_ratio: 1,
      component_count: 2,
      simplified_component_count: 1,
    });
    expect(first.reuse).toEqual({
      prepared_geometry: false,
      component_results: 0,
      total_components: 2,
    });

    const secondMessages: string[] = [];
    const second = await compiler.compileGlb(file, {
      decompose: { threshold: 0.2 },
      onProgress: ({ message }) => { if (message) secondMessages.push(message); },
    });
    expect(worker.configs).toEqual([
      { threshold: 0.1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 127 },
      { threshold: 1, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 1 },
      { threshold: 0.2, mctsNodes: 8, mctsIteration: 40, mctsMaxDepth: 2, maxChVertex: 4, maxConvexHull: 119 },
    ]);
    expect(secondMessages).toContain("Reusing prepared triangle geometry");
    expect(secondMessages.some((message) => message.includes("1 reused"))).toBe(true);
    expect(second.reuse).toEqual({
      prepared_geometry: true,
      component_results: 1,
      total_components: 2,
    });
    expect(second.report.config.effective).toMatchObject({
      max_hulls: 120,
      max_hulls_ceiling: 128,
    });
    expect(second.report.config.effective?.detail_budget_ratio).toBeCloseTo(0.94);
    compiler.terminate();
  });

  it("does not let callers mutate cached component results", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const file = new Blob([makeGlb()]);
    const first = await compiler.compileGlb(file);
    first.hulls[0].vertices[0] = 99;
    first.hulls[0].indices[0] = 2;

    const second = await compiler.compileGlb(file);

    expect(second.reuse.component_results).toBe(second.reuse.total_components);
    expect(second.hulls[0].vertices[0]).toBe(0);
    expect(second.hulls[0].indices[0]).toBe(0);
    compiler.terminate();
  });

  it("limits coarse thresholds for a scene-dominant connected body", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeScaledTetrahedraGlb(), {
      decompose: { threshold: 0.6 },
    });
    const detailed = worker.configs.find((config) => config.threshold !== 1);

    expect(detailed?.threshold).toBeGreaterThanOrEqual(0.1);
    expect(detailed?.threshold).toBeLessThan(0.22);
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_IMPORTANCE_GUARD",
      context: expect.objectContaining({ requested_threshold: 0.6 }),
    }));
    expect(result.report.config.effective).toMatchObject({
      important_component_max_threshold: 0.14,
      importance_guarded_component_count: 1,
    });
    compiler.terminate();
  });

  it("bounds recent component configurations while retaining nearby detail results", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const file = new Blob([makeScaledTetrahedraGlb()]);
    for (const threshold of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) {
      await compiler.compileGlb(file, {
        decompose: { threshold },
        componentPolicy: { enabled: false },
      });
    }
    expect(worker.configs).toHaveLength(14);

    const recent = await compiler.compileGlb(file, {
      decompose: { threshold: 0.7 },
      componentPolicy: { enabled: false },
    });
    expect(worker.configs).toHaveLength(14);
    expect(recent.reuse.component_results).toBe(2);

    const expired = await compiler.compileGlb(file, {
      decompose: { threshold: 0.1 },
      componentPolicy: { enabled: false },
    });
    expect(worker.configs).toHaveLength(16);
    expect(expired.reuse.component_results).toBe(0);
    compiler.terminate();
  });

  it("limits coarse settings on low-occupancy shells instead of filling their interiors", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeThinOpenTrayGlb(), {
      decompose: { threshold: 0.58 },
    });

    expect(worker.configs).toEqual([{
      threshold: 0.05,
      mctsNodes: 8,
      mctsIteration: 40,
      mctsMaxDepth: 2,
      maxChVertex: 16,
      maxConvexHull: 93,
    }]);
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HOLLOW_SHELL_GUARD",
      context: expect.objectContaining({
        requested_threshold: 0.58,
        effective_hollow_shell_threshold: 0.05,
      }),
    }));
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
    }));
    expect(result.report.config.effective).toMatchObject({
      hollow_shell_component_count: 1,
      guarded_hollow_shell_component_count: 1,
      hollow_shell_threshold: 0.05,
      hollow_shell_min_hulls: 8,
      adaptive_hull_vertices: true,
      max_hulls: 93,
      max_hulls_ceiling: 128,
      detail_budget_coarse_ratio: 0.7,
      effective_component_hull_vertices_min: 16,
      effective_component_hull_vertices_max: 16,
    });
    expect(result.report.config.effective?.detail_budget_ratio).toBeCloseTo(0.712);
    compiler.terminate();
  });

  it("rejects an explicit hull budget that cannot satisfy hollow-shell reservations", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(makeThinOpenTrayGlb(), {
      decompose: { threshold: 0.58 },
      componentPolicy: { maxHulls: 7 },
    })).rejects.toMatchObject({
      code: "INVALID_CONFIG",
      stage: "validating-input",
      context: {
        hollow_shell_component_count: 1,
        max_hulls: 7,
        required_minimum_hulls: 8,
      },
    });
    compiler.terminate();
  });

  it("assigns hull vertices from both geometric roundness and scene-relative size", async () => {
    const worker = new CompletingWorker();
    const compiler = compilerWith(() => worker);
    const result = await compiler.compileGlb(makeAdaptiveHullBudgetGlb());
    const caps = worker.configs.map((config) => config.maxChVertex as number);

    expect(caps).toHaveLength(3);
    expect(caps[0]).toBeGreaterThan(caps[1]);
    expect(caps[1]).toBeGreaterThan(caps[2]);
    expect(result.report.warnings).toContainEqual(expect.objectContaining({
      code: "INTERACTIVE_HULL_VERTICES_ADAPTED",
    }));
    expect(result.report.config.effective).toMatchObject({
      adaptive_hull_vertices: true,
      hull_vertex_roundness_metric: "isoperimetric_quotient",
      effective_component_hull_vertices_min: caps[2],
      effective_component_hull_vertices_max: caps[0],
    });
    compiler.terminate();
  });
});
