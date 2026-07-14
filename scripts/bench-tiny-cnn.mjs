import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4180);
const batches = (process.env.BENCH_BATCHES ?? "1,2,4,8,16,32,64")
  .split(",")
  .map(Number)
  .filter((size) => Number.isInteger(size) && size > 0);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);
const layouts = (process.env.BENCH_LAYOUTS ?? "nchw,nhwc")
  .split(",")
  .filter((layout) => layout === "nchw" || layout === "nhwc");

if (batches.length === 0) throw new Error("BENCH_BATCHES contains no valid sizes");
if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}
if (layouts.length === 0) {
  throw new Error("BENCH_LAYOUTS must contain nchw and/or nhwc");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () =>
      typeof globalThis.webnnPlayground?.benchmarkTinyCnnLayout === "function",
  );
  for (const inputLayout of layouts) {
    const reports = [];
    for (const batch of batches) {
      reports.push(
        await page.evaluate(
          ([layout, batchSize, warmupCount, iterationCount]) =>
            globalThis.webnnPlayground.benchmarkTinyCnnLayout(
              layout,
              batchSize,
              warmupCount,
              iterationCount,
            ),
          [inputLayout, batch, warmup, iterations],
        ),
      );
    }

    console.log(
      `\nChrome ${browserVersion} Canary headless / Tiny CNN / ${inputLayout.toUpperCase()} / WebNN npu / ` +
        `warmup=${warmup} iterations=${iterations}`,
    );
    console.log(
      "| batch | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error | probability sum error | matches |",
    );
    console.log(
      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    );
    for (const report of reports) {
      const cpuP50 = percentile(report.cpuSamplesMs, 0.5);
      const webnnP50 = percentile(report.webnnSamplesMs, 0.5);
      const webnnP95 = percentile(report.webnnSamplesMs, 0.95);
      console.log(
        `| ${report.batchSize} | ${report.operationsPerSample} | ` +
          `${fixed(cpuP50)} | ${fixed(webnnP50)} | ${fixed(webnnP95)} | ` +
          `${fixed(cpuP50 / webnnP50, 2)}x | ${fixed(report.contextMs)} | ` +
          `${fixed(report.graphMs)} | ${fixed(report.compileMs)} | ` +
          `${fixed(report.prepareMs)} | ${report.maxAbsError.toExponential(2)} | ` +
          `${report.maxProbabilitySumError.toExponential(2)} | ` +
          `${report.predictionMatches}/${report.batchSize} |`,
      );
    }
  }
});
