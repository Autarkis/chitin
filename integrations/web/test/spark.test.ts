import { SplatMesh } from "@sparkjsdev/spark";
import { readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it, vi } from "vitest";

import { loadSparkAsset } from "../src/spark.js";

const FIXTURE = resolve(__dirname, "conformance", "static_hull.phys");

function loadFixture(): ArrayBuffer {
  const buffer = readFileSync(FIXTURE);
  return buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  );
}

describe("Spark adapter", () => {
  it("pairs a supplied SplatMesh with parsed physics and shadow geometry", async () => {
    const visual = new SplatMesh();
    await visual.initialized;
    visual.position.set(2, 3, 4);
    const asset = await loadSparkAsset({
      visual,
      phys: loadFixture(),
      shadows: { mode: "collision-lod" },
    });

    expect(asset.root.children).toContain(visual);
    expect(asset.root.children).toContain(asset.shadowProxy);
    expect(asset.phys.hulls).toHaveLength(1);
    expect(asset.bounds.physics).not.toBeNull();
    expect(asset.root.position.toArray()).toEqual([2, 3, 4]);
    expect(visual.position.toArray()).toEqual([0, 0, 0]);

    asset.dispose();
    // Supplied visuals stay caller-owned by default.
    expect(visual.parent).toBeNull();
    expect(visual.position.toArray()).toEqual([2, 3, 4]);
    visual.dispose();
  });

  it("loads .phys URLs through an injected fetch implementation", async () => {
    const fetcher = vi.fn(async () => new Response(loadFixture()));
    const asset = await loadSparkAsset({
      visual: {},
      phys: "https://example.test/chair.phys",
      fetch: fetcher as typeof fetch,
    });

    expect(fetcher).toHaveBeenCalledWith("https://example.test/chair.phys", {
      signal: undefined,
    });
    expect(asset.shadowProxy).toBeNull();
    asset.dispose();
  });

  it("disposes an owned visual when sidecar parsing fails", async () => {
    const dispose = vi.spyOn(SplatMesh.prototype, "dispose");
    await expect(
      loadSparkAsset({ visual: {}, phys: new ArrayBuffer(4) }),
    ).rejects.toThrow(/file too small/);
    expect(dispose).toHaveBeenCalled();
    dispose.mockRestore();
  });

  it("fails before allocating a visual when already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      loadSparkAsset({
        visual: {},
        phys: loadFixture(),
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
