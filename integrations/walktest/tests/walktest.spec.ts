import { readFileSync } from "fs";
import { resolve } from "path";
import { test, expect } from "@playwright/test";

const PHYS_FILE = process.env.WALKTEST_PHYS ?? "";
const GRID = parseInt(process.env.WALKTEST_GRID ?? "4", 10);
const STEPS_PER_WAYPOINT = parseInt(process.env.WALKTEST_STEPS ?? "120", 10);
const FALL_THROUGH_TOLERANCE = 2.0;

interface WalkFrame {
  step: number;
  x: number;
  y: number;
  z: number;
  vy: number;
  grounded: boolean;
  stuck: boolean;
}

test.describe("walktest", () => {
  test.skip(!PHYS_FILE, "set WALKTEST_PHYS=/path/to/file.phys to run");

  test("capsule does not fall through collision geometry", async ({ page }) => {
    const physPath = resolve(PHYS_FILE);
    const physBuffer = readFileSync(physPath);
    const physBase64 = physBuffer.toString("base64");

    await page.goto("http://localhost:3219/");

    await page.waitForFunction(() => (window as any).__walktest?.ready, null, {
      timeout: 15_000,
    });

    await page.evaluate(async (b64: string) => {
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      await (window as any).__walktest.loadPhys(url);
      URL.revokeObjectURL(url);
    }, physBase64);

    const bounds = await page.evaluate(
      () => (window as any).__walktest.getSceneBounds()
    );
    expect(bounds).not.toBeNull();

    const { min, max } = bounds as {
      min: [number, number, number];
      max: [number, number, number];
    };

    const waypoints: [number, number, number][] = [];
    for (let xi = 0; xi < GRID; xi++) {
      for (let zi = 0; zi < GRID; zi++) {
        const x = min[0] + ((max[0] - min[0]) * (xi + 0.5)) / GRID;
        const z = min[2] + ((max[2] - min[2]) * (zi + 0.5)) / GRID;
        const y = max[1] + 2;
        waypoints.push([x, y, z]);
      }
    }

    const frames: WalkFrame[] = await page.evaluate(
      async ({
        wps,
        steps,
      }: {
        wps: [number, number, number][];
        steps: number;
      }) => {
        return (window as any).__walktest.runPlaybook(wps, steps);
      },
      { wps: waypoints, steps: STEPS_PER_WAYPOINT }
    );

    expect(frames.length).toBeGreaterThan(0);

    const floorY = min[1] - FALL_THROUGH_TOLERANCE;
    const fallThroughs = frames.filter((f) => f.y < floorY);
    const stuckFrames = frames.filter((f) => f.stuck);

    const fallRate = fallThroughs.length / frames.length;
    const stuckRate = stuckFrames.length / frames.length;

    console.log(`frames: ${frames.length}`);
    console.log(
      `fall-throughs: ${fallThroughs.length} (${(fallRate * 100).toFixed(1)}%)`
    );
    console.log(
      `stuck frames: ${stuckFrames.length} (${(stuckRate * 100).toFixed(1)}%)`
    );

    const report = {
      phys_file: PHYS_FILE,
      grid: GRID,
      steps_per_waypoint: STEPS_PER_WAYPOINT,
      total_frames: frames.length,
      fall_throughs: fallThroughs.length,
      fall_rate: Math.round(fallRate * 10000) / 10000,
      stuck_frames: stuckFrames.length,
      stuck_rate: Math.round(stuckRate * 10000) / 10000,
      scene_bounds: { min, max },
    };
    console.log("report:", JSON.stringify(report, null, 2));

    expect(fallRate).toBeLessThan(0.05);
  });
});
