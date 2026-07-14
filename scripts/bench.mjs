import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4174);
const sizes = (process.env.BENCH_SIZES ?? "32,64,128,256")
  .split(",")
  .map(Number)
  .filter((size) => Number.isInteger(size) && size > 0);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);

if (sizes.length === 0) throw new Error("BENCH_SIZES contains no valid sizes");
if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () => typeof globalThis.webnnPlayground?.benchmarkMatmul === "function",
  );
  const reports = [];
  for (const size of sizes) {
    reports.push(
      await page.evaluate(
        ([matrixSize, warmupCount, iterationCount]) =>
          globalThis.webnnPlayground.benchmarkMatmul(
            matrixSize,
            warmupCount,
            iterationCount,
          ),
        [size, warmup, iterations],
      ),
    );
  }

  console.log(
    `\nChrome ${browserVersion} Canary headless / ` +
      `WebNN ${reports[0]?.device ?? "unknown"} / ` +
      `warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| size | ops/sample | CPU p50 ms | WebNN fresh p50 ms | WebNN reuse p50 ms | reuse p95 ms | speedup | context ms | compile ms | prepare ms | max error |",
  );
  console.log(
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const cpuP50 = percentile(report.cpuSamplesMs, 0.5);
    const freshP50 = percentile(report.webnnFreshSamplesMs, 0.5);
    const webnnP50 = percentile(report.webnnSamplesMs, 0.5);
    const webnnP95 = percentile(report.webnnSamplesMs, 0.95);
    console.log(
      `| ${report.size} | ${report.batchSize} | ${fixed(cpuP50)} | ` +
        `${fixed(freshP50)} | ${fixed(webnnP50)} | ${fixed(webnnP95)} | ` +
        `${fixed(cpuP50 / webnnP50, 2)}x | ${fixed(report.contextMs)} | ` +
        `${fixed(report.compileMs)} | ${fixed(report.prepareMs)} | ` +
        `${report.maxAbsError.toExponential(2)} |`,
    );
  }
});
