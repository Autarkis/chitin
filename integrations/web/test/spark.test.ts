import { SplatMesh, dyno } from "@sparkjsdev/spark";
import { readFileSync } from "fs";
import { resolve } from "path";
import {
  BasicShadowMap,
  DirectionalLight,
  PCFShadowMap,
  type WebGLRenderer,
} from "three";
import { describe, expect, it, vi } from "vitest";

import { attachSparkDirectionalShadow, loadSparkAsset } from "../src/spark.js";

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

  it("composes and disposes a directional shadow receiver", async () => {
    const visual = new SplatMesh();
    await visual.initialized;
    const existing = dyno.dynoBlock(
      { gsplat: dyno.Gsplat },
      { gsplat: dyno.Gsplat },
      ({ gsplat }) => ({ gsplat }),
    );
    const onFrame = vi.fn();
    visual.worldModifiers = [existing];
    visual.onFrame = onFrame;
    const light = new DirectionalLight();
    light.castShadow = true;
    const renderer = rendererWithShadowMap(PCFShadowMap);
    const version = visual.version;

    const receiver = attachSparkDirectionalShadow({
      visual,
      renderer,
      light,
      dynamic: false,
    });

    expect(visual.worldModifiers).toHaveLength(2);
    expect(visual.worldModifiers).toContain(existing);
    expect(visual.generatorDirty).toBe(true);
    expect(visual.version).toBeGreaterThan(version);
    expect(receiver.strength).toBe(0.65);
    expect(receiver.bias).toBeNull();
    expect(receiver.dynamic).toBe(false);
    expect(visual.enableWorldToView).toBe(true);

    receiver.strength = 0.8;
    receiver.bias = -0.0002;
    receiver.dynamic = true;
    expect(receiver.strength).toBe(0.8);
    expect(receiver.bias).toBe(-0.0002);
    expect(receiver.dynamic).toBe(true);
    expect(() => {
      receiver.strength = 2;
    }).toThrow(/between 0 and 1/);
    expect(() => {
      receiver.bias = Number.NaN;
    }).toThrow(/finite or null/);

    const secondLight = new DirectionalLight();
    secondLight.castShadow = true;
    const secondReceiver = attachSparkDirectionalShadow({
      visual,
      renderer,
      light: secondLight,
      dynamic: false,
    });
    expect(visual.worldModifiers).toHaveLength(3);

    receiver.dispose();
    expect(visual.worldModifiers).toHaveLength(2);
    expect(visual.enableWorldToView).toBe(true);
    expect(visual.onFrame).not.toBe(onFrame);
    secondReceiver.dispose();
    expect(visual.worldModifiers).toEqual([existing]);
    expect(visual.onFrame).toBe(onFrame);
    expect(visual.enableWorldToView).toBe(false);
    receiver.dispose();
    visual.dispose();
  });

  it("rejects incompatible Three and Spark shadow configurations", async () => {
    const visual = new SplatMesh();
    await visual.initialized;
    const light = new DirectionalLight();

    expect(() =>
      attachSparkDirectionalShadow({
        visual,
        renderer: rendererWithShadowMap(PCFShadowMap, false),
        light,
      }),
    ).toThrow(/shadowMap.enabled/);

    expect(() =>
      attachSparkDirectionalShadow({
        visual,
        renderer: rendererWithShadowMap(BasicShadowMap),
        light,
      }),
    ).toThrow(/PCFShadowMap/);

    expect(() =>
      attachSparkDirectionalShadow({
        visual,
        renderer: rendererWithShadowMap(PCFShadowMap),
        light,
      }),
    ).toThrow(/must cast shadows/);
    visual.dispose();
  });
});

function rendererWithShadowMap(
  type: typeof PCFShadowMap | typeof BasicShadowMap,
  enabled = true,
): WebGLRenderer {
  return { shadowMap: { enabled, type } } as WebGLRenderer;
}
