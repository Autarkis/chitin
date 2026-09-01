import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import type { ChitinDemoApi } from "../src/demo-api.js";
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
  const preview = await page.evaluate(() => window.__chitinDemo.previewAvailable);
  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.busy).toBe(false);
  expect(state.hulls).toBe(1);
  expect(state.verdict).toBe("not_evaluated");
  expect(state.reportVersion).toBe(1);
  expect(state.appliedThreshold).toBe(0.1);
  expect(state.qualityMeasured).toBe(false);
  if (preview) {
    expect(state.colliderRevealCount).toBe(1);
    await expect(page.locator(".viewport-panel")).toHaveAttribute("data-collider-reveal", "complete");
  }

  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.locator("#report-panel")).toBeVisible();
  await expect(page.locator("#report-status")).toHaveText("Complete");
  await expect(page.locator("#report-profile")).toHaveText("Interactive");
  await expect(page.locator("#report-verdict")).toHaveText("Not evaluated");
  await expect(page.locator("#report-checks")).toContainText(
    "Profile acceptance checks were not run.",
  );
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
  const preview = await page.evaluate(() => window.__chitinDemo.previewAvailable);
  if (preview) {
    const previewState = await page.evaluate(() => window.__chitinDemo.state());
    expect(previewState.sourceMeshes).toBeGreaterThan(0);
    expect(previewState.sourceFilledMeshes).toBe(previewState.sourceMeshes);
  }
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

  const preview = await page.evaluate(() => window.__chitinDemo.previewAvailable);
  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.hulls).toBe(0);
  if (preview) {
    expect(state.sourceVisible).toBe(true);
    expect(state.colliderVisible).toBe(false);
    expect(state.sourceMeshes).toBeGreaterThan(0);
  }
});

test("replacing an artifact resets report and applied-detail state", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-wicker").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.getByRole("button", { name: "Hide report" })).toBeVisible();

  await page.locator("#file-input").setInputFiles({
    name: "open-panels.glb",
    mimeType: "model/gltf-binary",
    buffer: Buffer.from(makeGlb()),
  });

  await expect(page.locator("#error-code")).toContainText("GEOMETRY REPAIR NEEDED");
  await expect(page.locator("#report-button")).toHaveText("View report");
  await expect(page.locator("#report-output")).toBeHidden();
  expect((await page.evaluate(() => window.__chitinDemo.state())).appliedThreshold).toBeNull();
});

test("preview layers and responsive controls remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  const preview = await page.evaluate(() => window.__chitinDemo.previewAvailable);
  test.skip(!preview, "WebGL unavailable");
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
  const preview = await page.evaluate(() => window.__chitinDemo.previewAvailable);
  expect(pendingState.busy).toBe(true);
  expect(pendingState.hulls).toBe(appliedState.hulls);
  if (preview) {
    expect(pendingState.colliderVisible).toBe(true);
  }

  await expect(page.locator("#threshold-status")).toContainText("Applied 0.30", { timeout: 60_000 });
  await expect(page.locator("#result-summary")).toContainText("Detail 0.30 applied");
  const state = await page.evaluate(() => window.__chitinDemo.state());
  expect(state.appliedThreshold).toBe(0.3);
});

test("fit presets apply named detail budgets", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-wicker").click();
  await expect(page.locator("#threshold-status")).toContainText("Applied 0.10", {
    timeout: 60_000,
  });

  const gameProp = page.getByRole("button", { name: /Game prop/i });
  await gameProp.click();
  await expect(gameProp).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#threshold")).toHaveValue("0.25");
  await expect(page.locator("#threshold-status")).toContainText("Applied 0.25", {
    timeout: 60_000,
  });
  expect((await page.evaluate(() => window.__chitinDemo.state())).appliedThreshold).toBe(0.25);
});

test("quality diagnostics measure fit on demand", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-wicker").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#quality-metrics")).toBeHidden();

  await page.getByRole("button", { name: "Run quality diagnostics" }).click();
  await expect.poll(
    async () => (await page.evaluate(() => window.__chitinDemo.state())).qualityMeasured,
    { timeout: 60_000 },
  ).toBe(true);
  await expect(page.locator("#quality-metrics")).toBeVisible();
  await expect(page.locator("#surface-coverage")).toHaveText("100.0%");
  await expect(page.locator("#volume-precision")).toHaveText("100.0%");
  await expect(page.locator("#false-fill")).toHaveText("0.0%");
  await expect(page.locator("#measurement-note")).toContainText("Sampled locally");
  await expect(page.getByRole("button", { name: "Rerun quality diagnostics" })).toBeEnabled();
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
    "hollow-shell guard",
  );
  await page.getByRole("button", { name: "View report" }).click();
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_HOLLOW_SHELL_GUARD");
  await expect(page.locator("#report-output")).toContainText("INTERACTIVE_HULL_VERTICES_ADAPTED");
  await expect(page.locator("#report-output")).toContainText('"hollow_shell_threshold": 0.05');
});

test("profile selector recompiles with walkable and robotics diagnostics", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);

  const interactive = page.locator('[data-profile="interactive"]');
  const walkable = page.locator('[data-profile="walkable"]');
  const robotics = page.locator('[data-profile="robotics"]');

  await expect(interactive).toHaveAttribute("aria-pressed", "true");
  await expect(interactive).toBeEnabled();
  await expect(walkable).toBeEnabled();
  await expect(robotics).toBeEnabled();
  await walkable.click();
  await expect(walkable).toHaveAttribute("aria-pressed", "true");
  expect((await page.evaluate(() => window.__chitinDemo.state())).profile).toBe("walkable");
  await robotics.click();
  await expect(robotics).toHaveAttribute("aria-pressed", "true");
  expect((await page.evaluate(() => window.__chitinDemo.state())).profile).toBe("robotics");
});

test("result exposes output size, per-hull visibility, and matching copyable code", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.getByTestId("sample-fish").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#output-size")).not.toHaveText("—");
  const hullToggles = page.locator("#hull-list input");
  await expect(hullToggles).toHaveCount((await page.evaluate(() => window.__chitinDemo.state())).hulls);
  await hullToggles.first().uncheck();
  expect((await page.evaluate(() => window.__chitinDemo.state())).visibleHulls)
    .toBe((await page.evaluate(() => window.__chitinDemo.state())).hulls - 1);
  await page.locator('[data-profile="walkable"]').click();
  await expect(page.locator("#code-snippet")).toContainText('profile: "walkable"');
  await page.locator("#copy-snippet").click();
  await expect(page.locator("#copy-status")).toContainText(/Copied|Clipboard unavailable/);
});

test("invalid input is actionable and a cancelled compile can be replaced", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.locator("#file-input").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("nope") });
  await expect(page.locator("#error-message")).toContainText("binary glTF");
  await expect(page.locator("#error-suggestion")).toContainText("GLB 2.0");
  await page.getByTestId("sample-fish").click();
  await expect(page.locator("#cancel-button")).toBeEnabled();
  await page.locator("#cancel-button").click();
  await expect(page.locator("#progress-copy")).toContainText("Cancelled");
  await page.getByTestId("sample-wicker").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#file-name")).toHaveText("clearcoat-wicker.glb");
});

test("drag-and-drop compiles while the UI event loop remains responsive", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);
  await page.evaluate(() => {
    (window as any).__heartbeat = 0;
    window.setInterval(() => (window as any).__heartbeat++, 10);
  });
  await page.evaluate(async () => {
    const response = await fetch("./assets/barramundi-fish.glb");
    const file = new File([await response.arrayBuffer()], "dropped-fish.glb", { type: "model/gltf-binary" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    window.dispatchEvent(new DragEvent("dragenter", { bubbles: true, dataTransfer: transfer }));
    window.dispatchEvent(new DragEvent("drop", { bubbles: true, dataTransfer: transfer }));
  });
  await expect(page.locator("#file-name")).toHaveText("dropped-fish.glb");
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });
  expect(await page.evaluate(() => (window as any).__heartbeat)).toBeGreaterThan(2);
});

test("rapier simulation runs after compilation", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    let frameTime = 0;
    window.requestAnimationFrame = (callback) => window.setTimeout(() => callback(frameTime += 8), 8);
    window.cancelAnimationFrame = (handle) => window.clearTimeout(handle);
  });
  await page.goto("/");
  await page.waitForFunction(() => window.__chitinDemo?.ready);

  await page.getByTestId("sample-wicker").click();
  await expect(page.getByText("Artifact compiled")).toBeVisible({ timeout: 60_000 });

  const simButton = page.locator("#simulate-button");
  await expect(simButton).toBeVisible();
  await simButton.click();

  await expect.poll(
    async () => (await page.evaluate(() => window.__chitinDemo.state())).simulationActive,
    { timeout: 30_000 },
  ).toBe(true);

  const initialHeight = (await page.evaluate(() => window.__chitinDemo.state())).simulationHeight;
  expect(initialHeight).not.toBeNull();
  await expect.poll(
    async () => (await page.evaluate(() => window.__chitinDemo.state())).simulationHeight ?? Infinity,
  ).toBeLessThan(initialHeight! - 0.001);

  await expect(page.locator("#show-simulation")).toBeChecked();
  await expect(page.locator("#simulate-status")).toContainText("Sphere dropped");
  await expect(simButton).toHaveText("Restart simulation");

  expect(errors).toEqual([]);
});

declare global {
  interface Window {
    __chitinDemo: ChitinDemoApi;
  }
}
