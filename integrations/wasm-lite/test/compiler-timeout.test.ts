import { describe, it, expect } from "vitest";
import { ChitinCompiler, compileMesh } from "../src/compiler.js";
import { ChitinError } from "../src/errors.js";
import type { WorkerRequest, WorkerResponse } from "../src/worker-protocol.js";
import {
  CompletingWorker,
  HoldingWorker,
  compilerWith,
  HULLS,
} from "./compiler-fixture.js";
import { makeGlb } from "./glb-fixture.js";

describe("timeout", () => {
  it("rejects with TIMEOUT when deadline expires mid-decompose", async () => {
    // HoldingWorker never completes, so the timeout fires
    const compiler = compilerWith(() => new HoldingWorker());
    const promise = compiler.compileGlb(makeGlb(), {
      timeout: 50,
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    await expect(promise).rejects.toMatchObject({
      code: "TIMEOUT",
      retryable: true,
    });
    expect(promise).rejects.toBeInstanceOf(ChitinError);
    compiler.terminate();
  });

  it("does not fire on fast completion", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const result = await compiler.compileGlb(makeGlb(), {
      timeout: 5000,
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    expect(result.phys).toBeInstanceOf(ArrayBuffer);
    compiler.terminate();
  });

  it("recovers after timeout for the next compilation", async () => {
    let callCount = 0;
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm", version: "0.2.0" },
      workerFactory: () => {
        callCount++;
        // First call: holding worker (will timeout)
        // Second call: completing worker (will succeed)
        return callCount === 1 ? new HoldingWorker() : new CompletingWorker();
      },
      maxWorkers: 1,
    });

    // First call times out
    await expect(
      compiler.compileGlb(makeGlb(), {
        timeout: 50,
        checkManifold: false,
        componentPolicy: { enabled: false },
      }),
    ).rejects.toMatchObject({ code: "TIMEOUT" });

    // Second call succeeds
    const result = await compiler.compileGlb(makeGlb(), {
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    expect(result.phys).toBeInstanceOf(ArrayBuffer);
    compiler.terminate();
  });

  it("prefers CANCELLED over TIMEOUT when caller signal fires first", async () => {
    const controller = new AbortController();
    const compiler = compilerWith(() => new HoldingWorker());
    const promise = compiler.compileGlb(makeGlb(), {
      timeout: 5000,
      signal: controller.signal,
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ code: "CANCELLED" });
    compiler.terminate();
  });

  it("preserves CANCELLED when terminate aborts a timed compilation", async () => {
    const compiler = compilerWith(() => new HoldingWorker());
    const promise = compiler.compileGlb(makeGlb(), {
      timeout: 5000,
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    compiler.terminate();
    await expect(promise).rejects.toMatchObject({ code: "CANCELLED" });
  });
});

describe("compileMesh", () => {
  it("compiles raw vertex/face arrays", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const vertices = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const faces = new Int32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
    const result = await compiler.compileMesh(vertices, faces, {
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    expect(result.phys).toBeInstanceOf(ArrayBuffer);
    expect(result.report.report_version).toBe(1);
    expect(result.source.vertex_count).toBe(4);
    expect(result.source.triangle_count).toBe(4);
    expect(result.source.mesh_count).toBe(1);
    compiler.terminate();
  });

  it("one-shot compileMesh works", async () => {
    const vertices = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const faces = new Int32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
    const result = await compileMesh(vertices, faces, {
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm", version: "0.2.0" },
      workerFactory: () => new CompletingWorker(),
      maxWorkers: 1,
      checkManifold: false,
      componentPolicy: { enabled: false },
    });
    expect(result.phys).toBeInstanceOf(ArrayBuffer);
    expect(result.report.status).toBe("complete");
  });

  it("emits progress events", async () => {
    const compiler = compilerWith(() => new CompletingWorker());
    const vertices = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const faces = new Int32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
    const stages: string[] = [];
    await compiler.compileMesh(vertices, faces, {
      checkManifold: false,
      componentPolicy: { enabled: false },
      onProgress: (p) => stages.push(p.stage),
    });
    // Should start with parsing-input (no reading-input for raw arrays)
    expect(stages[0]).toBe("parsing-input");
    expect(stages).toContain("decomposing");
    expect(stages).toContain("done");
    // Should NOT contain reading-input
    expect(stages).not.toContain("reading-input");
    compiler.terminate();
  });
});
