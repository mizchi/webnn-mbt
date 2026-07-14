import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4175);
const batches = (process.env.BENCH_BATCHES ?? "1,16,100")
  .split(",")
  .map(Number)
  .filter((size) => Number.isInteger(size) && size > 0);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);

if (batches.length === 0) throw new Error("BENCH_BATCHES contains no valid sizes");
if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () => typeof globalThis.webnnPlayground?.benchmarkMnist === "function",
  );
  const reports = [];
  for (const batch of batches) {
    reports.push(
      await page.evaluate(
        ([batchSize, warmupCount, iterationCount]) =>
          globalThis.webnnPlayground.benchmarkMnist(
            batchSize,
            warmupCount,
            iterationCount,
          ),
        [batch, warmup, iterations],
      ),
    );
  }

  console.log(
    `\nChrome ${browserVersion} Canary headless / MNIST MLP / WebNN npu / ` +
      `warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| batch | ops/sample | CPU p50 ms | WebNN fresh p50 ms | WebNN reuse p50 ms | reuse p95 ms | speedup | fixture ms | context ms | graph ms | compile ms | prepare ms | max error | correct |",
  );
  console.log(
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const cpuP50 = percentile(report.cpuSamplesMs, 0.5);
    const freshP50 = percentile(report.webnnFreshSamplesMs, 0.5);
    const webnnP50 = percentile(report.webnnSamplesMs, 0.5);
    const webnnP95 = percentile(report.webnnSamplesMs, 0.95);
    console.log(
      `| ${report.batchSize} | ${report.operationsPerSample} | ` +
        `${fixed(cpuP50)} | ${fixed(freshP50)} | ${fixed(webnnP50)} | ` +
        `${fixed(webnnP95)} | ${fixed(cpuP50 / webnnP50, 2)}x | ` +
        `${fixed(report.fixtureMs)} | ${fixed(report.contextMs)} | ` +
        `${fixed(report.graphMs)} | ${fixed(report.compileMs)} | ` +
        `${fixed(report.prepareMs)} | ${report.maxAbsError.toExponential(2)} | ` +
        `${report.webnnCorrect}/${report.batchSize} |`,
    );
  }
});
