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
  projects: playwrightDefaults.browsers.map((device) => ({
    name: device.split(" ").pop()!.toLowerCase(),
    use: { ...devices[device] },
  })),
  webServer: {
    command: "npm run preview",
    port: 4179,
    reuseExistingServer: true,
  },
});
