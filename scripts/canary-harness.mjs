import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";

const browserChannel =
  process.env.WEBNN_BROWSER_CHANNEL ?? "chrome-canary";

async function waitForServer(url) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

export async function withCanaryPage({ port }, operation) {
  const baseURL = `http://127.0.0.1:${port}`;
  const server = spawn(process.execPath, ["scripts/serve.mjs"], {
    cwd: process.cwd(),
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "ignore", "inherit"],
  });
  let browser;
  try {
    await waitForServer(baseURL);
    browser = await chromium.launch({
      channel: browserChannel,
      headless: true,
      args: ["--enable-features=WebMachineLearningNeuralNetwork"],
    });
    const page = await browser.newPage();
    await page.goto(baseURL);
    return await operation({
      page,
      browserVersion: browser.version(),
      baseURL,
    });
  } finally {
    await browser?.close();
    server.kill("SIGTERM");
  }
}

export function percentile(samples, fraction) {
  const sorted = [...samples].sort((a, b) => a - b);
  const index = Math.ceil(fraction * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

export function fixed(value, digits = 3) {
  return Number(value).toFixed(digits);
}
