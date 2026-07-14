import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

async function sourceFiles(directory = "src") {
  const entries = await readdir(new URL(`${directory}/`, root), {
    withFileTypes: true,
  });
  const files = [];
  for (const entry of entries) {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...(await sourceFiles(path)));
    } else if (/\.(?:mbt|mbti)$/.test(entry.name) || entry.name === "moon.pkg") {
      files.push(path);
    }
  }
  return files;
}

test("MoonBit module and repository use their canonical identities", async () => {
  const module = await read("moon.mod");
  const packageJson = JSON.parse(await read("package.json"));

  assert.match(module, /^name = "mizchi\/webnn"$/m);
  assert.match(
    module,
    /^repository = "https:\/\/github\.com\/mizchi\/webnn-mbt"$/m,
  );
  assert.doesNotMatch(module, /playground/i);
  assert.equal(packageJson.name, "webnn-mbt");
  assert.equal(packageJson.private, true);
});

test("published sources do not retain the playground module path", async () => {
  const files = [
    ...(await sourceFiles()),
    "README.md",
    "public/index.html",
    "scripts/prepare-bert-fixture.mjs",
    "scripts/serve.mjs",
  ];

  for (const file of files) {
    assert.doesNotMatch(
      await read(file),
      /webnn-playground/,
      `${file} still contains the old project identity`,
    );
    assert.doesNotMatch(
      await read(file),
      /mizchi\/webnn-mbt\//,
      `${file} uses the repository name as a MoonBit module path`,
    );
  }
});
