import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("Spark demo lights stored sidecars through deferred hulls", async ({
  page,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto("http://localhost:3219/spark-demo.html");

  await expect(page.locator("#status")).toContainText("Live", {
    timeout: 60_000,
  });
  await expect(page.locator("#splat-count")).toHaveText(/^\d[\d,]+$/);
  await expect(page.locator("#receiver-name")).toHaveText("Spark robot head");
  await expect(page.locator("#hull-count")).toHaveText("2");
  await expect(page.locator("#receiver-hull-count")).toHaveText(/^\d+$/);
  expect(
    Number(await page.locator("#receiver-hull-count").textContent()),
  ).toBeGreaterThan(0);
  await expect(page.locator("canvas")).toBeVisible();
  await expect(page.locator("#generated-robot-phys")).toHaveCount(0);

  const receiverHulls = page.locator("#receiver-hulls");
  const receiverSplats = page.locator("#receiver-splats");
  const lighting = page.locator("#lighting");
  const selfShadows = page.locator("#self-shadows");
  const ambientOcclusion = page.locator("#ambient-occlusion");
  await expect(receiverHulls).toBeChecked();
  await expect(receiverSplats).toBeChecked();
  await expect(lighting).toBeChecked();
  await expect(selfShadows).toBeChecked();
  await expect(ambientOcclusion).toBeChecked();
  await expect(page.locator("#ao-strength")).toHaveValue("0.35");
  await expect(page.locator("#environment")).toHaveValue("studio");
  await expect(page.locator("#environment option")).toHaveCount(3);
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-environment",
    "studio",
  );
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-lighting-mode",
    "deferred-hulls",
  );
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-self-shadow-source",
    "phys-hulls",
  );
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-hull-binding-status",
    "applied",
  );

  const sceneTab = page.getByRole("tab", { name: "Scene" });
  const setupTab = page.getByRole("tab", { name: "Chitin setup" });
  await sceneTab.press("ArrowRight");
  await expect(setupTab).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#hull-setup-panel")).toBeVisible();
  await expect(page.locator("#phys-file-name")).toHaveText(
    "robot-head.phys",
  );
  await expect(page.locator("#setup-hull-count")).toHaveText(/^[1-9]\d*$/);
  await expect(page.locator("#phys-version")).toHaveText(/^v\d+$/);
  await expect(page.locator("#phys-size")).toHaveText(/KB$/);
  await expect(page.locator("#binding-status")).toContainText(
    "applied to deferred lighting",
  );
  await expect(page.locator("#apply-hulls")).toBeDisabled();
  await expect(page.locator("#detail")).toHaveValue("0.025");
  await expect(page.locator("#reconstruction-resolution")).toHaveValue("72");
  await expect(page.locator("#max-hulls")).toHaveValue("96");
  await expect(page.locator("#download-phys")).toBeEnabled();

  await page.getByRole("button", { name: "Balanced" }).click();
  expect(await page.locator("#detail").inputValue()).toMatch(/^0\.10?$/);
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-hull-detail",
    "0.100",
  );
  await expect(page.locator("#explode-distance-control")).toBeHidden();
  await page.locator("#explode-hulls").check();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-receiver-hulls-exploded",
    "true",
  );
  await expect(page.locator("#explode-distance-control")).toBeVisible();
  await page.locator("#explode-hulls").uncheck();

  await expect(page.locator("#compile-hulls")).toBeEnabled();
  await page.locator(".advanced-settings summary").click();
  await expect(page.locator("#reconstruction-resolution option")).toHaveCount(5);
  await expect(page.locator("#max-hulls option")).toHaveCount(5);

  await page.locator("#phys-file-input").setInputFiles(
    resolve("harness/assets/robot-head.phys"),
  );
  await expect(page.locator("#binding-status")).toContainText("staged");
  await expect(page.locator("#apply-hulls")).toBeEnabled();
  await page.getByLabel("Hull position X").fill("0.10");
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-hull-binding-status",
    "staged",
  );
  await page.locator("#apply-hulls").click();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-hull-binding-status",
    "applied",
  );
  expect(
    JSON.parse(
      (await page
        .locator("#viewport")
        .getAttribute("data-hull-binding-transform"))!,
    ),
  ).toEqual({ position: [0.1, 0, 0], rotation: [0, 0, 0], scale: 1 });
  await expect(page.locator("#apply-hulls")).toBeDisabled();
  await page.locator("#reset-hulls").click();
  await page.locator("#apply-hulls").click();

  await receiverSplats.uncheck();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-receiver-splats-visible",
    "false",
  );
  await receiverHulls.uncheck();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-receiver-hulls-visible",
    "false",
  );
  await receiverSplats.check();
  await receiverHulls.check();

  await page.getByRole("tab", { name: "Scene" }).click();
  await lighting.uncheck();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-lighting",
    "false",
  );
  await expect(page.locator("#strength")).toBeDisabled();
  await expect(page.locator("#ao-strength")).toBeDisabled();
  await expect(page.locator("#status")).toContainText(
    "real-time lighting is off",
  );
  await lighting.check();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-lighting",
    "true",
  );
  await expect(page.locator("#strength")).toBeEnabled();
  await expect(page.locator("#ao-strength")).toBeEnabled();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-external-shadow-strength",
    "0.750",
  );
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-self-shadow-strength",
    "0.750",
  );
  await selfShadows.uncheck();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-self-shadows",
    "false",
  );
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-self-shadow-strength",
    "0.000",
  );
  await ambientOcclusion.uncheck();
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-ambient-occlusion",
    "false",
  );
  await expect(page.locator("#ao-strength")).toBeDisabled();
  expect(browserErrors).toEqual([]);
});
