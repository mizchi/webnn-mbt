import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
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

test("mizchi/webnn provides a root facade consumed by the example", async () => {
  const packageManifest = await read("src/moon.pkg");
  const facade = await read("src/top.mbt");
  const exampleManifest = await read("examples/playground/src/app/moon.pkg");

  assert.match(packageManifest, /"mizchi\/webnn\/backend\/webnn" @backend/);
  assert.match(facade, /pub using @backend/);
  assert.match(facade, /type WebNNGraphBuilder/);
  assert.match(facade, /type WebNNRuntime/);
  assert.match(exampleManifest, /"mizchi\/webnn",/);
  assert.doesNotMatch(exampleManifest, /"mizchi\/webnn\/backend\/webnn"/);
});

test("application-only benchmark and MNIST packages live in the example", async () => {
  assert.equal(await exists("src/benchmark"), false);
  assert.equal(await exists("src/mnist"), false);
  assert.equal(await exists("examples/playground/src/benchmark/moon.pkg"), true);
  assert.equal(await exists("examples/playground/src/mnist/moon.pkg"), true);
});

test("playground API responsibilities are split across focused files", async () => {
  const files = await readdir(new URL("examples/playground/src/app/", root));
  const expose = await read("examples/playground/src/app/expose.mbt");
  for (const name of [
    "expose.mbt",
    "core_api.mbt",
    "concurrency_api.mbt",
    "benchmark_api.mbt",
    "tflite_basic_api.mbt",
    "tflite_cache_api.mbt",
    "tflite_fixture_api.mbt",
    "tflite_runner_api.mbt",
    "main.mbt",
  ]) {
    assert.ok(files.includes(name), `${name} must exist`);
  }
  assert.ok((await read("examples/playground/src/app/main.mbt")).split("\n").length < 40);
  assert.doesNotMatch(expose, /^extern "js" fn expose_api\(/m);
  for (const registration of [
    "expose_core_api",
    "expose_tflite_basic_api",
    "expose_tflite_runtime_api",
    "expose_benchmark_api",
    "expose_mnist_and_concurrency_api",
  ]) {
    assert.match(expose, new RegExp(`extern "js" fn ${registration}\\(`));
  }
});

test("validated public structs keep their representations private", async () => {
  const contracts = [
    ["src/shape/shape.mbt", "Shape"],
    ["src/litert/lowering.mbt", "LiteRtModel"],
    ["src/litert/lowering.mbt", "LiteRtLoweredValue"],
    ["src/litert/lowering.mbt", "LiteRtValue"],
    ["src/litert/tflite.mbt", "TfliteRuntimeModel"],
    ["src/litert/tflite.mbt", "TfliteRuntimeTensor"],
    ["src/backend/webnn/litert.mbt", "WebNNLoweredLiteRtGraph"],
    ["examples/playground/src/mnist/fixture.mbt", "Fixture"],
  ];

  for (const [path, type] of contracts) {
    const source = await read(path);
    assert.match(source, new RegExp(`pub struct ${type}(?:\\[| \\{)`));
    assert.doesNotMatch(source, new RegExp(`pub\\(all\\) struct ${type}`));
  }
});
