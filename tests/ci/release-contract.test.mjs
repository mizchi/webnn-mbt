import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

test("v0.1.0 release metadata is ready", async () => {
  const moduleManifest = await read("moon.mod");
  const changelog = await read("CHANGELOG.md");

  assert.match(moduleManifest, /name = "mizchi\/webnn"/);
  assert.match(moduleManifest, /version = "0\.1\.0"/);
  assert.match(moduleManifest, /license = "MIT"/);
  assert.match(moduleManifest, /repository = "https:\/\/github\.com\/mizchi\/webnn-mbt"/);
  assert.match(changelog, /^## \[0\.1\.0\] - 2026-07-15$/m);
});

test("published module excludes repository-only assets", async () => {
  const moduleManifest = await read("moon.mod");

  for (const path of [
    "examples",
    "fixtures",
    "public",
    "scripts",
    "tests",
    "justfile",
    "moon.work",
    "package.json",
    "playwright.config.ts",
    "pnpm-lock.yaml",
    "src/**/*_wbtest.mbt",
  ]) {
    assert.match(moduleManifest, new RegExp(`"${path.replaceAll("*", "\\*")}"`));
  }
});
