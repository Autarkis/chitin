import {
  DirectionalLight,
  PerspectiveCamera,
  Scene,
  Vector2,
  type WebGLRenderer,
} from "three";
import { describe, expect, it } from "vitest";

import { createDeferredHullLightingPass } from "../src/deferred.js";

function rendererStub(): WebGLRenderer {
  return {
    getDrawingBufferSize(target: Vector2) {
      return target.set(640, 360);
    },
  } as WebGLRenderer;
}

describe("deferred hull lighting", () => {
  it("exposes validated mutable light controls", () => {
    const light = new DirectionalLight();
    light.castShadow = true;
    const pass = createDeferredHullLightingPass({
      renderer: rendererStub(),
      scene: new Scene(),
      camera: new PerspectiveCamera(),
      lights: [{ light, strength: 0.65, bias: null, normalBias: 0.02 }],
    });

    expect(pass.lights).toHaveLength(1);
    expect(pass.lights[0].strength).toBe(0.65);
    expect(pass.lights[0].bias).toBeNull();
    expect(pass.lights[0].normalBias).toBe(0.02);

    pass.lights[0].strength = 0.8;
    pass.lights[0].bias = -0.0002;
    pass.lights[0].normalBias = 0.01;
    expect(pass.lights[0].strength).toBe(0.8);
    expect(pass.lights[0].bias).toBe(-0.0002);
    expect(pass.lights[0].normalBias).toBe(0.01);
    expect(() => (pass.lights[0].strength = 2)).toThrow(/between 0 and 1/);
    expect(() => (pass.lights[0].bias = Number.NaN)).toThrow(/finite or null/);
    expect(() => (pass.lights[0].normalBias = -1)).toThrow(/non-negative/);
    pass.dispose();
  });

  it("rejects missing lights, invalid layers, and non-shadow lights", () => {
    const options = {
      renderer: rendererStub(),
      scene: new Scene(),
      camera: new PerspectiveCamera(),
    };
    expect(() =>
      createDeferredHullLightingPass({ ...options, lights: [] }),
    ).toThrow(/at least one light/);

    const light = new DirectionalLight();
    expect(() =>
      createDeferredHullLightingPass({
        ...options,
        hullLayer: 32,
        lights: [{ light, strength: 0.5 }],
      }),
    ).toThrow(/hullLayer/);
    expect(() =>
      createDeferredHullLightingPass({
        ...options,
        lights: [{ light, strength: 0.5 }],
      }),
    ).toThrow(/must cast shadows/);
  });
});
