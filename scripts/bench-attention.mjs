import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4184);
const widths = (process.env.BENCH_WIDTHS ?? "32,64,128,256")
  .split(",")
  .map(Number)
  .filter((width) => Number.isInteger(width) && width > 0);
const tokens = Number(process.env.BENCH_TOKENS ?? 16);
const heads = Number(process.env.BENCH_HEADS ?? 4);
const masks = (process.env.BENCH_MASKS ?? "causal")
  .split(",")
  .map((mask) => mask.trim())
  .filter(Boolean);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);
const supportedMasks = new Set([
  "none",
  "causal",
  "padding",
  "causal-padding",
]);

if (widths.length === 0) {
  throw new Error("BENCH_WIDTHS contains no valid widths");
}
if (!Number.isInteger(tokens) || tokens <= 0) {
  throw new Error("BENCH_TOKENS must be a positive integer");
}
if (!Number.isInteger(heads) || heads <= 0) {
  throw new Error("BENCH_HEADS must be a positive integer");
}
for (const width of widths) {
  if (width % heads !== 0) {
    throw new Error(`width ${width} must be divisible by heads ${heads}`);
  }
}
if (masks.length === 0 || masks.some((mask) => !supportedMasks.has(mask))) {
  throw new Error(
    "BENCH_MASKS must contain none, causal, padding, or causal-padding",
  );
}
if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () =>
      typeof globalThis.webnnPlayground?.benchmarkSelfAttention ===
      "function",
  );
  const reports = [];
  for (const mask of masks) {
    for (const width of widths) {
      reports.push(
        await page.evaluate(
          ([tokenCount, modelWidth, headCount, maskMode, warmupCount, iterationCount]) =>
            globalThis.webnnPlayground.benchmarkSelfAttention(
              tokenCount,
              modelWidth,
              headCount,
              maskMode,
              warmupCount,
              iterationCount,
            ),
          [tokens, width, heads, mask, warmup, iterations],
        ),
      );
    }
  }

  console.log(
    `\nChrome ${browserVersion} Canary headless / Self-attention / ` +
      `tokens=${tokens} heads=${heads} WebNN npu / ` +
      `warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| width | heads | mask | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error |",
  );
  console.log(
    "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const cpuP50 = percentile(report.cpuSamplesMs, 0.5);
    const webnnP50 = percentile(report.webnnSamplesMs, 0.5);
    const webnnP95 = percentile(report.webnnSamplesMs, 0.95);
    console.log(
      `| ${report.width} | ${report.heads} | ${report.maskMode} | ` +
        `${report.operationsPerSample} | ${fixed(cpuP50)} | ` +
        `${fixed(webnnP50)} | ${fixed(webnnP95)} | ` +
        `${fixed(cpuP50 / webnnP50, 2)}x | ${fixed(report.contextMs)} | ` +
        `${fixed(report.graphMs)} | ${fixed(report.compileMs)} | ` +
        `${fixed(report.prepareMs)} | ${report.maxAbsError.toExponential(2)} |`,
    );
  }
});
