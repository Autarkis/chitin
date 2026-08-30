import { afterEach, describe, expect, it, vi } from "vitest";

import { CompletingWorker, compilerWith } from "./compiler-fixture.js";
import { makeGlb } from "./glb-fixture.js";

describe("ChitinCompiler inputs", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("compiles an ArrayBuffer and returns .phys, source facts, progress, and the v1 report", async () => {
    const stages: string[] = [];
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      onProgress: ({ stage }) => stages.push(stage),
      decompose: { threshold: 0.05 },
    });
    compiler.terminate();

    expect(new DataView(result.phys).getUint32(0, true)).toBe(0x53594850);
    expect(result.hulls).toHaveLength(2);
    expect(result.source).toEqual({
      mesh_count: 2,
      primitive_count: 4,
      node_count: 2,
      vertex_count: 12,
      triangle_count: 4,
    });
    expect(stages.slice(0, 4)).toEqual([
      "reading-input", "parsing-input", "validating-input", "loading-wasm",
    ]);
    expect(stages.filter((stage) => stage === "decomposing")).toHaveLength(5);
    expect(stages.slice(-2)).toEqual(["writing-phys", "done"]);
    expect(result.report).toMatchObject({
      report_version: 1,
      profile: "interactive",
      verdict: { status: "not_evaluated" },
      input: { kind: "glb", source_vertices: 12, processed_vertices: 6, mesh_vertices: 6 },
      runtime: {
        kind: "browser_wasm",
        version: "0.2.0",
        dependencies: { "@autarkis/chitin-wasm": "0.2.0" },
      },
      reproducibility: { scope: "same_runtime_toolchain", deterministic: null },
      config: { requested: { threshold: 0.05 } },
    });
    expect(result.report.reproducibility.artifact_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("measures artifact fit only when quality sampling is requested", async () => {
    const stages: string[] = [];
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      checkManifold: false,
      quality: { surfaceSamples: 32, volumeSamples: 64, minColliderSamples: 1 },
      onProgress: ({ stage }) => stages.push(stage),
    });
    compiler.terminate();

    expect(stages).toContain("verifying");
    expect(result.report.verdict.status).toBe("not_evaluated");
    expect(result.report.metrics.quality_method).toEqual({
      value: "deterministic_halton_v1",
      unit: "method",
      status: "measured",
    });
    expect(result.report.metrics.source_surface_coverage.status).toBe("measured");
    expect(result.report.timings_ms.verify).toBeGreaterThanOrEqual(0);
  });

  it("accepts Blob and ArrayBufferView inputs", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb(new Blob([makeGlb()]))).resolves.toMatchObject({
      source: { triangle_count: 4 },
    });
    const bytes = new Uint8Array(makeGlb());
    await expect(compiler.compileGlb(bytes.subarray(0))).resolves.toMatchObject({
      source: { vertex_count: 12 },
    });
    compiler.terminate();
  });

  it("fetches URL inputs and records the source artifact", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(makeGlb(), { status: 200 })));
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(new URL("https://example.test/chair.glb"));
    expect(fetch).toHaveBeenCalledWith(
      "https://example.test/chair.glb",
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(result.report.artifacts.source).toBe("https://example.test/chair.glb");
    compiler.terminate();
  });

  it("returns a structured LOAD_ERROR for failed URL inputs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 404 })));
    const compiler = compilerWith(() => new CompletingWorker());
    await expect(compiler.compileGlb("https://example.test/missing.glb")).rejects.toMatchObject({
      code: "LOAD_ERROR",
      stage: "reading-input",
      retryable: false,
      context: { http_status: 404 },
    });
    compiler.terminate();
  });

  it("terminate cancels a compilation while its URL input is loading", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
          }),
      ),
    );
    const compiler = compilerWith(() => new CompletingWorker());
    const pending = compiler.compileGlb("https://example.test/slow.glb");
    compiler.terminate();
    await expect(pending).rejects.toMatchObject({
      code: "CANCELLED",
      stage: "reading-input",
      retryable: true,
    });
  });
});
