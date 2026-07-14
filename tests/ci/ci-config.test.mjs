import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

test("GitHub Actions runs the MoonBit and WebNN contracts", async () => {
  const workflow = await read(".github/workflows/ci.yml");

  assert.match(workflow, /runs-on: ubuntu-24\.04/);
  assert.match(workflow, /node-version: 24/);
  assert.match(workflow, /version: 10\.33\.0/);
  assert.match(workflow, /pnpm install --frozen-lockfile/);
  assert.match(workflow, /just ci-unit/);
  assert.match(workflow, /just ci-e2e/);
  assert.match(
    workflow,
    /WEBNN_BROWSER_CHANNEL: chromium-tip-of-tree/,
  );
  assert.match(
    workflow,
    /playwright install --with-deps --only-shell chromium-tip-of-tree/,
  );
  assert.match(workflow, /just-version: 1\.50\.0/);

  const actionRefs = [...workflow.matchAll(/^\s*uses:\s*([^\s#]+)/gm)].map(
    ([, ref]) => ref,
  );
  assert.ok(actionRefs.length > 0, "workflow must use setup actions");
  for (const ref of actionRefs) {
    assert.match(ref, /@[0-9a-f]{40}$/, `action must be SHA-pinned: ${ref}`);
  }
});

test("CI tasks are portable and formatting is non-mutating", async () => {
  const justfile = await read("justfile");

  assert.match(justfile, /set shell := \["bash", "-cu"\]/);
  assert.match(justfile, /^ci-unit:/m);
  assert.match(justfile, /moon fmt --check/);
  assert.match(justfile, /^ci-e2e:/m);
});

test("Playwright selects Canary for Testing through the environment", async () => {
  const config = await read("playwright.config.ts");
  const harness = await read("scripts/canary-harness.mjs");

  assert.match(config, /process\.env\.WEBNN_BROWSER_CHANNEL/);
  assert.match(config, /\?\? "chrome-canary"/);
  assert.match(config, /--enable-features=WebMachineLearningNeuralNetwork/);
  assert.match(harness, /process\.env\.WEBNN_BROWSER_CHANNEL/);
  assert.match(harness, /channel: browserChannel/);
});
