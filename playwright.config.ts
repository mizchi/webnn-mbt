import { defineConfig } from "@playwright/test";

const browserChannel =
  process.env.WEBNN_BROWSER_CHANNEL ?? "chrome-canary";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: {
    timeout: 10_000,
  },
  reporter: process.env.CI
    ? [["line"], ["html", { open: "never" }]]
    : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: browserChannel,
    headless: true,
    launchOptions: {
      args: ["--enable-features=WebMachineLearningNeuralNetwork"],
    },
    trace: "retain-on-failure",
  },
  webServer: {
    command: "pnpm run build && pnpm run serve",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
  },
});
