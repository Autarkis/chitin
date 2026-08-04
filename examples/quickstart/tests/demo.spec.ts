import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { makeGlb, makeThinOpenTrayGlb } from "../../../integrations/wasm-lite/test/glb-fixture.js";

test("opening index.html directly explains how to start the runtime", async ({ page }) => {
  await page.goto(pathToFileURL(resolve("index.html")).href);

  await expect(page.getByRole("heading", { name: "Start the local runtime." })).toBeVisible();
  await expect(page.getByText("START_DEMO.cmd")).toBeVisible();
  await expect(page.locator("#app")).toBeHidden();
});

test("built-in GLB compiles into a downloadable collider artifact", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);

  await expect(page.getByRole("heading", { name: /Make geometry ready to collide/i })).toBeVisible();
  await expect(page.getByText("Runtime ready")).toBeVisible();
  await page.getByTestId("sample-wicker").click();

  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#source-triangles")).toHaveText("3,072");
  await expect(page.locator("#collider-triangles")).not.toHaveText("—");
  await expect(page.locator("#triangle-ratio")).toContainText("%");
  await expect(page.locator("#hull-count")).not.toHaveText("—");
  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.busy).toBe(false);
  expect(state.hulls).toBe(1);
  expect(state.verdict).toBe("not_evaluated");
  expect(state.reportVersion).toBe(1);
  expect(state.appliedThreshold).toBe(0.1);
  expect(state.colliderRevealCount).toBe(1);
  await expect(page.locator(".viewport-panel")).toHaveAttribute("data-collider-reveal", "complete");

  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.locator("#report-output")).toContainText('"report_version": 1');
  await expect(page.locator("#report-output")).toContainText('"status": "not_evaluated"');

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: /Download .phys/i }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("clearcoat-wicker.phys");
  expect(errors).toEqual([]);
});

test("verified lightweight real-asset fixtures compile from bundled GLBs", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);

  await page.getByTestId("sample-fish").click();
  await expect(page.locator("#file-name")).toHaveText("barramundi-fish.glb");
  await expect(page.locator("#source-triangles")).toHaveText("3,864", { timeout: 60_000 });
  const fishHullCount = Number(await page.locator("#hull-count").textContent());
  expect(fishHullCount).toBeGreaterThanOrEqual(3);
  expect(fishHullCount).toBeLessThanOrEqual(9);
  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED");

  await page.locator("#threshold").evaluate((element) => {
    const slider = element as HTMLInputElement;
    slider.value = "0.60";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator("#threshold-status")).toContainText(
    "scale-aware body detail active",
    { timeout: 60_000 },
  );
  const coarseFishHullCount = Number(await page.locator("#hull-count").textContent());
  expect(coarseFishHullCount).toBeGreaterThanOrEqual(5);
  expect(coarseFishHullCount).toBeLessThanOrEqual(fishHullCount);
  await expect(page.locator("#result-time")).toContainText("2/3 parts reused");
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_IMPORTANCE_GUARD");
  const previewState = await page.evaluate(() => window.__chitinDemo.state());
  expect(previewState.sourceMeshes).toBeGreaterThan(0);
  expect(previewState.sourceFilledMeshes).toBe(previewState.sourceMeshes);
  expect(consoleErrors).toEqual([]);
});

test("open GLBs keep their source preview and explain the full-compiler repair path", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.locator("#file-input").setInputFiles({
    name: "open-panels.glb",
    mimeType: "model/gltf-binary",
    buffer: Buffer.from(makeGlb()),
  });

  await expect(page.locator("#error-code")).toContainText("GEOMETRY REPAIR NEEDED");
  await expect(page.locator("#error-message")).toContainText(
    "Connected part 1 of 2 is not a closed solid",
  );
  await expect(page.locator("#error-message")).toContainText("3 boundary edges");
  await expect(page.locator("#error-suggestion")).toContainText("full Chitin compiler");
  await expect(page.getByRole("button", { name: "Choose another GLB" })).toBeVisible();
  await expect(page.locator("#threshold-status")).toContainText("no collider produced");

  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.hulls).toBe(0);
  expect(state.sourceVisible).toBe(true);
  expect(state.colliderVisible).toBe(false);
  expect(state.sourceMeshes).toBeGreaterThan(0);
});

test("preview layers and responsive controls remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await expect(page.locator("#viewport")).toBeVisible();
  await expect(page.locator(".control-panel")).toBeVisible();
  await expect(page.locator("#file-button")).toBeVisible();
  await expect(page.getByTestId("sample-wicker")).toBeVisible();
  await expect(page.getByTestId("sample-dish")).toBeVisible();

  await page.getByTestId("sample-wicker").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });

  const sourceToggle = page.getByLabel("Source");
  const colliderToggle = page.getByLabel("Colliders");
  await colliderToggle.uncheck();
  await expect(sourceToggle).toBeChecked();
  await expect(colliderToggle).not.toBeChecked();
  const sourceOnlyState = await page.evaluate(() => window.__chitinDemo.state());
  expect(sourceOnlyState.sourceVisible).toBe(true);
  expect(sourceOnlyState.colliderVisible).toBe(false);
  expect(sourceOnlyState.sourceMeshes).toBeGreaterThan(0);
  expect(sourceOnlyState.sourceFilledMeshes).toBe(sourceOnlyState.sourceMeshes);

  await colliderToggle.check();
  const explodeToggle = page.getByRole("checkbox", { name: "Explode", exact: true });
  await expect(explodeToggle).toBeEnabled();
  await explodeToggle.check();
  await expect(page.getByLabel("Exploded view separation")).toBeEnabled();
  await expect.poll(async () => (await page.evaluate(() => window.__chitinDemo.state())).explosionAmount)
    .toBeGreaterThan(0.25);
  expect((await page.evaluate(() => window.__chitinDemo.state())).exploded).toBe(true);
  await explodeToggle.uncheck();
  await expect.poll(async () => (await page.evaluate(() => window.__chitinDemo.state())).explosionAmount)
    .toBeLessThan(0.01);
});

test("detail changes automatically recompile at the displayed setting", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-wicker").click();
  await expect(page.locator("#threshold-status")).toContainText("Applied 0.10", { timeout: 60_000 });
  const appliedState = await page.evaluate(() => window.__chitinDemo.state());

  const pendingState = await page.locator("#threshold").evaluate((element) => {
    const slider = element as HTMLInputElement;
    slider.value = "0.30";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
    return window.__chitinDemo.state();
  });
  expect(pendingState.busy).toBe(true);
  expect(pendingState.hulls).toBe(appliedState.hulls);
  expect(pendingState.colliderVisible).toBe(true);

  await expect(page.locator("#threshold-status")).toContainText("Applied 0.30", { timeout: 60_000 });
  await expect(page.locator("#result-summary")).toContainText("Detail 0.30 applied");
  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.appliedThreshold).toBe(0.3);
});

test("detail changes reduce complexity for decomposable geometry", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-fish").click();
  await expect.poll(
    async () => (await page.evaluate(() => window.__chitinDemo.state())).appliedThreshold,
    { timeout: 60_000 },
  ).toBe(0.1);
  const fine = {
    hulls: Number((await page.locator("#hull-count").textContent())?.replaceAll(",", "")),
    triangles: Number((await page.locator("#collider-triangles").textContent())?.replaceAll(",", "")),
  };

  await page.locator("#threshold").evaluate((element) => {
    const slider = element as HTMLInputElement;
    slider.value = "0.60";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect.poll(
    async () => (await page.evaluate(() => window.__chitinDemo.state())).appliedThreshold,
    { timeout: 60_000 },
  ).toBe(0.6);
  const coarse = {
    hulls: Number((await page.locator("#hull-count").textContent())?.replaceAll(",", "")),
    triangles: Number((await page.locator("#collider-triangles").textContent())?.replaceAll(",", "")),
  };

  expect(coarse.hulls).toBeLessThanOrEqual(fine.hulls);
  expect(coarse.triangles).toBeLessThanOrEqual(fine.triangles);
  expect(coarse.hulls < fine.hulls || coarse.triangles < fine.triangles).toBe(true);
});

test("coarse requests expose the hollow-shell guard instead of silently filling interiors", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.locator("#threshold").evaluate((element) => {
    const slider = element as HTMLInputElement;
    slider.value = "0.58";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator("#file-input").setInputFiles({
    name: "thin-open-tray.glb",
    mimeType: "model/gltf-binary",
    buffer: Buffer.from(makeThinOpenTrayGlb()),
  });

  await expect(page.locator("#threshold-status")).toContainText(
    "hollow-shell guard active",
    { timeout: 60_000 },
  );
  await expect(page.locator("#result-summary")).toContainText(
    "hollow-shell guard + adaptive hull detail",
  );
  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_HOLLOW_SHELL_GUARD");
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_HULL_VERTICES_ADAPTED");
  await expect(page.locator("#report-output")).toContainText('"hollow_shell_threshold": 0.05');
});

declare global {
  interface Window {
    __chitinDemo: {
      ready: boolean;
      state(): {
        busy: boolean;
        hulls: number;
        verdict: string | null;
        reportVersion: number | null;
        appliedThreshold: number | null;
        sourceVisible: boolean;
        colliderVisible: boolean;
        sourceMeshes: number;
        sourceFilledMeshes: number;
        colliderRevealActive: boolean;
        colliderRevealCount: number;
        exploded: boolean;
        explosionAmount: number;
        reusedComponents: number;
      };
    };
  }
}
