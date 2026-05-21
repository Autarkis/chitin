import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    headless: true,
  },
  webServer: {
    command: "npx serve harness -l 3219 --no-clipboard",
    port: 3219,
    reuseExistingServer: true,
  },
});
