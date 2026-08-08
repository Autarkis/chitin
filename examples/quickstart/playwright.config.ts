import { defineConfig, devices } from "@playwright/test";
import { playwrightDefaults } from "../../scripts/playwright-base.mjs";

export default defineConfig({
  testDir: "./tests",
  timeout: playwrightDefaults.timeout,
  // CoACD jobs contend for CPU and WASM memory; serialize for stable timings.
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4179",
    headless: playwrightDefaults.headless,
  },
  // Firefox excluded: headless Firefox lacks software WebGL.
  // WASM correctness on Firefox is gated by integrations/walktest.
  projects: playwrightDefaults.browsers
    .filter((device) => !device.includes("Firefox"))
    .map((device) => ({
      name: device.split(" ").pop()!.toLowerCase(),
      use: { ...devices[device] },
    })),
  webServer: {
    command: "npm run preview",
    port: 4179,
    reuseExistingServer: true,
  },
});
