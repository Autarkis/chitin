import { expect, test } from "@playwright/test";

test("Spark shadow demo renders its compiled sidecar", async ({ page }) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto("http://localhost:3219/spark-demo.html");

  await expect(page.locator("#status")).toContainText("Live", {
    timeout: 90_000,
  });
  await expect(page.locator("#splat-count")).toHaveText(/^\d[\d,]+$/);
  await expect(page.locator("#receiver-name")).toHaveText(
    /Spark robot head|local fallback/,
  );
  await expect(page.locator("#hull-count")).toHaveText("2");
  await expect(page.locator("#receiver-hull-count")).toHaveText(/^\d+$/);
  expect(
    Number(await page.locator("#receiver-hull-count").textContent()),
  ).toBeGreaterThan(0);
  await expect(page.locator("canvas")).toBeVisible();

  const receiverHulls = page.locator("#receiver-hulls");
  const receiverSplats = page.locator("#receiver-splats");
  await expect(receiverHulls).toBeChecked();
  await expect(receiverSplats).toBeChecked();
  await expect(page.locator("#environment")).toHaveValue("studio");
  await expect(page.locator("#environment option")).toHaveCount(3);
  await expect(page.locator("#viewport")).toHaveAttribute(
    "data-environment",
    "studio",
  );
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
  expect(browserErrors).toEqual([]);
});
