import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4176);
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
    () => typeof globalThis.webnnPlayground?.benchmarkMnistCache === "function",
  );
  const reports = [];
  for (const batch of batches) {
    reports.push(
      await page.evaluate(
        ([batchSize, warmupCount, iterationCount]) =>
          globalThis.webnnPlayground.benchmarkMnistCache(
            batchSize,
            warmupCount,
            iterationCount,
          ),
        [batch, warmup, iterations],
      ),
    );
  }

  console.log(
    `\nChrome ${browserVersion} Canary headless / MNIST compiled program cache / ` +
      `WebNN npu / warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| batch | ops/sample | cold p50 ms | cold p95 ms | hit p50 ms | hit p95 ms | cold/hit | cache entries | max error | matches |",
  );
  console.log(
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const coldP50 = percentile(report.coldSamplesMs, 0.5);
    const coldP95 = percentile(report.coldSamplesMs, 0.95);
    const hitP50 = percentile(report.hitSamplesMs, 0.5);
    const hitP95 = percentile(report.hitSamplesMs, 0.95);
    console.log(
      `| ${report.batchSize} | ${report.operationsPerSample} | ` +
        `${fixed(coldP50)} | ${fixed(coldP95)} | ` +
        `${fixed(hitP50)} | ${fixed(hitP95)} | ` +
        `${fixed(coldP50 / hitP50, 2)}x | ${report.cacheSize} | ` +
        `${report.maxAbsError.toExponential(2)} | ` +
        `${report.predictionMatches}/${report.batchSize} |`,
    );
  }
});
