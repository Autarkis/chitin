import { describe, expect, it } from "vitest";

import { ChitinCompiler, compileGlb } from "../src/compiler.js";
import type { WorkerLike } from "../src/worker-client.js";
import type { WorkerRequest, WorkerResponse } from "../src/worker-protocol.js";
import {
  CompletingWorker,
  HULLS,
  HoldingWorker,
  compilerWith,
  internalCompile,
  makeScaledTetrahedraGlb,
  trackedSignal,
} from "./compiler-fixture.js";
import { makeGlb } from "./glb-fixture.js";

describe("ChitinCompiler lifecycle", () => {
  it("runs connected components through a bounded worker pool", async () => {
    let active = 0;
    let maximumActive = 0;
    class DelayedWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        active++;
        maximumActive = Math.max(maximumActive, active);
        setTimeout(() => {
          active--;
          this.onmessage?.({
            data: { type: "result", id: message.id, hulls: HULLS },
          } as MessageEvent<WorkerResponse>);
        }, 10);
      }
    }
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => new DelayedWorker(),
      maxWorkers: 2,
    });
    await compiler.compileGlb(makeGlb(), {
      componentPolicy: { enabled: false },
      checkManifold: false,
    });
    expect(maximumActive).toBe(2);
    compiler.terminate();
  });

  it("reports the first invalid component deterministically before parallel scheduling", async () => {
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => new CompletingWorker(),
      maxWorkers: 2,
    });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "NON_MANIFOLD",
      stage: "validating-input",
      context: { component_index: 0, component_number: 1, component_count: 2 },
    });
    compiler.terminate();
  });

  it("adds connected-part context and full-compiler guidance to topology failures", async () => {
    class RejectingWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        queueMicrotask(() => this.onmessage?.({ data: {
          type: "error",
          id: message.id,
          code: "NON_MANIFOLD",
          message: "boundary (open) edge; topology summary: 3 boundary edges",
          stage: null,
          suggestion: null,
          retryable: false,
          context: { boundary_edges: 3, non_manifold_edges: 0, degenerate_triangles: 0 },
        } } as MessageEvent<WorkerResponse>));
      }
    }
    const compiler = compilerWith(() => new RejectingWorker());
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "NON_MANIFOLD",
      stage: "validating-input",
      retryable: false,
      context: {
        component_index: 0,
        component_number: 1,
        component_count: 2,
        boundary_edges: 3,
      },
    });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      suggestion: expect.stringContaining("full Chitin compiler"),
    });
    compiler.terminate();
  });

  it("cancels an in-flight compile and can recover on a fresh worker", async () => {
    const workers: WorkerLike[] = [];
    let spawnSignalled: () => void;
    const spawnSignal = new Promise<void>((resolve) => {
      spawnSignalled = resolve;
    });
    class SignallingHoldingWorker extends HoldingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type === "decompose") spawnSignalled();
      }
    }
    const roles: Array<() => WorkerLike> = [
      () => new SignallingHoldingWorker(),
      () => new CompletingWorker(),
    ];
    const compiler = compilerWith(() => {
      const makeWorker = roles.shift();
      if (!makeWorker) throw new Error("test expected exactly two workers to be spawned");
      const worker = makeWorker();
      workers.push(worker);
      return worker;
    });
    const controller = new AbortController();
    const pending = compiler.compileGlb(makeGlb(), { signal: controller.signal });
    await spawnSignal;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
    await expect(compiler.compileGlb(makeGlb())).resolves.toMatchObject({
      source: { mesh_count: 2 },
    });
    expect(workers).toHaveLength(2);
    compiler.terminate();
  });

  it("rejects concurrent work immediately with COMPILER_BUSY", async () => {
    const compiler = compilerWith(() => new HoldingWorker());
    const controller = new AbortController();
    const pending = compiler.compileGlb(makeGlb(), { signal: controller.signal });
    await expect(compiler.compileGlb(makeGlb())).rejects.toMatchObject({
      code: "COMPILER_BUSY",
      retryable: true,
    });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
  });

  it("one-shot compileGlb always releases its worker", async () => {
    const worker = new CompletingWorker();
    await compileGlb(makeGlb(), {
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => worker,
      maxWorkers: 1,
    });
    expect(worker.terminated).toBe(true);
  });

  describe("abort-listener balance", () => {
    it("balances listeners on normal completion", async () => {
      const compiler = compilerWith(() => new CompletingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      await internalCompile(compiler, makeGlb(), { signal });

      expect(counts()).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });

    it("balances listeners when worker decomposition rejects", async () => {
      class RejectingWorker extends CompletingWorker {
        override postMessage(message: WorkerRequest): void {
          if (message.type !== "decompose") return;
          queueMicrotask(() => this.onmessage?.({
            data: {
              type: "error",
              id: message.id,
              code: "WORKER_ERROR",
              message: "simulated decompose failure",
              stage: null,
              suggestion: null,
              retryable: false,
              context: {},
            },
          } as MessageEvent<WorkerResponse>));
        }
      }
      const compiler = compilerWith(() => new RejectingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      await expect(internalCompile(compiler, makeGlb(), { signal })).rejects.toMatchObject({
        code: "WORKER_ERROR",
      });

      expect(counts()).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });

    it("balances listeners when the signal aborts mid-flight", async () => {
      const compiler = compilerWith(() => new HoldingWorker());
      const controller = new AbortController();
      const { signal, counts } = trackedSignal(controller);

      const pending = internalCompile(compiler, makeGlb(), { signal });
      await Promise.resolve();
      controller.abort();

      await expect(pending).rejects.toMatchObject({ code: "CANCELLED" });
      expect(counts()).toEqual({ adds: 1, removes: 1 });
      compiler.terminate();
    });
  });

  it("rewraps writePhys failures with their stage and cause", async () => {
    class EmptyHullWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        queueMicrotask(() => this.onmessage?.({
          data: {
            type: "result",
            id: message.id,
            hulls: [{ vertices: new Float32Array(0), indices: new Uint32Array(0) }],
          },
        } as MessageEvent<WorkerResponse>));
      }
    }
    const compiler = compilerWith(() => new EmptyHullWorker());

    const failure = await compiler.compileGlb(makeGlb()).catch((error: unknown) => error);
    compiler.terminate();

    expect(failure).toMatchObject({
      code: "INVALID_MESH",
      stage: "writing-phys",
      message: expect.stringContaining("empty hull"),
    });
    expect((failure as { cause?: unknown }).cause).toMatchObject({
      code: "INVALID_MESH",
      stage: null,
      message: expect.stringContaining("empty hull"),
    });
  });

  it("retains successful sibling cache entries after a concurrent failure", async () => {
    let worker1Attempts = 0;
    class FlakyWorker extends CompletingWorker {
      override postMessage(message: WorkerRequest): void {
        if (message.type !== "decompose") return;
        worker1Attempts++;
        if (worker1Attempts === 1) {
          queueMicrotask(() => this.onmessage?.({
            data: {
              type: "error",
              id: message.id,
              code: "WORKER_ERROR",
              message: "simulated first-attempt failure",
              stage: null,
              suggestion: null,
              retryable: false,
              context: {},
            },
          } as MessageEvent<WorkerResponse>));
          return;
        }
        super.postMessage(message);
      }
    }
    const worker0 = new CompletingWorker();
    const worker1 = new FlakyWorker();
    const worker1Respawn = new CompletingWorker();
    const workers = [worker0, worker1, worker1Respawn];
    let nextWorker = 0;
    const compiler = new ChitinCompiler({
      wasm: { js: "coacd.mjs", wasm: "coacd.wasm" },
      workerFactory: () => workers[nextWorker++],
      maxWorkers: 2,
    });
    const file = new Blob([makeScaledTetrahedraGlb()]);

    await expect(compiler.compileGlb(file, {
      componentPolicy: { enabled: false },
    })).rejects.toMatchObject({ code: "WORKER_ERROR" });
    expect(worker0.configs).toHaveLength(1);

    const recovered = await compiler.compileGlb(file, {
      componentPolicy: { enabled: false },
    });

    expect(worker0.configs).toHaveLength(1);
    expect(worker1.configs).toHaveLength(0);
    expect(worker1Respawn.configs).toHaveLength(1);
    expect(recovered.reuse.component_results).toBe(1);
    compiler.terminate();
  });
});
