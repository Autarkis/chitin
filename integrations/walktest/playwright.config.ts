import { defineConfig, devices } from "@playwright/test";
import { playwrightDefaults } from "../../scripts/playwright-base.mjs";

export default defineConfig({
  testDir: "./tests",
  timeout: playwrightDefaults.timeout,
  use: {
    headless: playwrightDefaults.headless,
  },
  projects: playwrightDefaults.browsers.map((device) => ({
    name: device.split(" ").pop()!.toLowerCase(),
    use: { ...devices[device] },
  })),
  webServer: {
    command: "npx serve harness -l 3219 --no-clipboard",
    port: 3219,
    reuseExistingServer: true,
  },
});
