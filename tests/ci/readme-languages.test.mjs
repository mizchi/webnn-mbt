import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

test("English and Japanese READMEs link to each other", async () => {
  const english = await read("README.md");
  const japanese = await read("README.ja.md");

  assert.match(english, /^# webnn-mbt\n\nEnglish \| \[日本語\]\(\.\/README\.ja\.md\)/);
  assert.match(japanese, /^# webnn-mbt\n\n\[English\]\(\.\/README\.md\) \| 日本語/);
});

test("README.md is English and README.ja.md is Japanese", async () => {
  const english = await read("README.md");
  const japanese = await read("README.ja.md");

  assert.match(english, /^## Requirements$/m);
  assert.match(english, /^## Workspace$/m);
  assert.match(english, /^## Design$/m);
  assert.match(english, /^## Limitations$/m);
  assert.doesNotMatch(english, /^## (必要な環境|設計|制限)$/m);

  assert.match(japanese, /^## 必要な環境$/m);
  assert.match(japanese, /^## ワークスペース$/m);
  assert.match(japanese, /^## 設計$/m);
  assert.match(japanese, /^## 制限$/m);
});

test("READMEs show a minimal inference using the root facade", async () => {
  const english = await read("README.md");
  const japanese = await read("README.ja.md");

  for (const readme of [english, japanese]) {
    assert.match(readme, /moon add mizchi\/webnn/);
    assert.match(readme, /"mizchi\/webnn"/);
    assert.match(readme, /@webnn\.WebNNGraphBuilder::new/);
    assert.match(readme, /@webnn\.DevicePreference::Npu/);
    assert.match(readme, /compile_program_single/);
    assert.match(readme, /defer program\.destroy\(\)/);
  }

  assert.match(english, /^## Quick start$/m);
  assert.match(japanese, /^## クイックスタート$/m);
});
