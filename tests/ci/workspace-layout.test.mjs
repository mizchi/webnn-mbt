import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

async function exists(path) {
  try {
    await access(new URL(path, root));
    return true;
  } catch {
    return false;
  }
}

test("workspace separates the library from the playground application", async () => {
  const workspace = await read("moon.work");
  const exampleModule = await read("examples/playground/moon.mod");
  const justfile = await read("justfile");
  const index = await read("public/index.html");
  const packageJson = JSON.parse(await read("package.json"));

  assert.match(workspace, /members = \[\s*"\."\s*,\s*"\.\/examples\/playground"\s*,?\s*\]/);
  assert.match(exampleModule, /^name = "mizchi\/webnn-examples"$/m);
  assert.match(exampleModule, /"mizchi\/webnn@0\.1\.0"/);
  assert.equal(await exists("src/app/main.mbt"), false);
  assert.equal(await exists("src/app/moon.pkg"), false);
  assert.equal(await exists("src/app/pkg.generated.mbti"), false);
  assert.equal(await exists("examples/playground/src/app/main.mbt"), true);
  assert.equal(await exists("examples/playground/src/app/moon.pkg"), true);
  assert.match(
    justfile,
    /^\s*moon build --target js --release examples\/playground\/src\/app$/m,
  );
  assert.equal(
    packageJson.scripts.build,
    "moon build --target js --release examples/playground/src/app",
  );
  assert.match(
    index,
    /\/_build\/js\/release\/build\/mizchi\/webnn-examples\/app\/app\.js/,
  );
});
