import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4177);
const pools = (process.env.BENCH_POOLS ?? "1,2,4,8")
  .split(",")
  .map(Number)
  .filter((size) => Number.isInteger(size) && size > 0);
const requests = Number(process.env.BENCH_REQUESTS ?? 256);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);

if (pools.length === 0) throw new Error("BENCH_POOLS contains no valid sizes");
if (!Number.isInteger(requests) || requests <= 0) {
  throw new Error("BENCH_REQUESTS must be a positive integer");
}
if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () => typeof globalThis.webnnPlayground?.benchmarkMnistPool === "function",
  );
  const reports = [];
  for (const pool of pools) {
    reports.push(
      await page.evaluate(
        ([poolSize, requestCount, warmupCount, iterationCount]) =>
          globalThis.webnnPlayground.benchmarkMnistPool(
            poolSize,
            requestCount,
            warmupCount,
            iterationCount,
          ),
        [pool, requests, warmup, iterations],
      ),
    );
  }

  const baseline = reports.find((report) => report.poolSize === 1) ?? reports[0];
  const baselineMs = percentile(baseline.samplesMs, 0.5);
  const comparisonLabel = baseline.poolSize === 1 ? "vs pool 1" : "vs first";
  console.log(
    `\nChrome ${browserVersion} Canary headless / MNIST execution pool / ` +
      `WebNN npu / concurrent requests=${requests} warmup=${warmup} ` +
      `iterations=${iterations}`,
  );
  console.log(
    `| pool | max concurrency | setup ms | group p50 ms | group p95 ms | ms/request | requests/s | ${comparisonLabel} | max error | matches |`,
  );
  console.log(
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const groupP50 = percentile(report.samplesMs, 0.5);
    const groupP95 = percentile(report.samplesMs, 0.95);
    console.log(
      `| ${report.poolSize} | ${report.maximumConcurrency} | ` +
        `${fixed(report.setupMs)} | ${fixed(groupP50)} | ` +
        `${fixed(groupP95)} | ${fixed(groupP50 / report.requestCount)} | ` +
        `${fixed((report.requestCount * 1000) / groupP50, 1)} | ` +
        `${fixed(baselineMs / groupP50, 2)}x | ` +
        `${report.maxAbsError.toExponential(2)} | ` +
        `${report.predictionMatches}/${report.requestCount} |`,
    );
  }
});
