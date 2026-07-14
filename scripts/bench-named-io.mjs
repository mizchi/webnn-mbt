import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4178);
const sizes = (process.env.BENCH_SIZES ?? "64,1024,16384")
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
    () => typeof globalThis.webnnPlayground?.benchmarkNamedIo === "function",
  );
  const reports = [];
  for (const size of sizes) {
    reports.push(
      await page.evaluate(
        ([valueCount, warmupCount, iterationCount]) =>
          globalThis.webnnPlayground.benchmarkNamedIo(
            valueCount,
            warmupCount,
            iterationCount,
          ),
        [size, warmup, iterations],
      ),
    );
  }

  console.log(
    `\nChrome ${browserVersion} Canary headless / named I/O readback / ` +
      `WebNN npu / warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| values | ops/sample | 1 output p50 ms | p95 ms | 2 outputs p50 ms | p95 ms | 2/1 | delta ms | compile 1/2 ms | prepare 1/2 ms | max error |",
  );
  console.log(
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const report of reports) {
    const oneP50 = percentile(report.oneOutputSamplesMs, 0.5);
    const oneP95 = percentile(report.oneOutputSamplesMs, 0.95);
    const twoP50 = percentile(report.twoOutputSamplesMs, 0.5);
    const twoP95 = percentile(report.twoOutputSamplesMs, 0.95);
    console.log(
      `| ${report.size} | ${report.operationsPerSample} | ` +
        `${fixed(oneP50)} | ${fixed(oneP95)} | ` +
        `${fixed(twoP50)} | ${fixed(twoP95)} | ` +
        `${fixed(twoP50 / oneP50, 2)}x | ${fixed(twoP50 - oneP50)} | ` +
        `${fixed(report.oneCompileMs)}/${fixed(report.twoCompileMs)} | ` +
        `${fixed(report.onePrepareMs)}/${fixed(report.twoPrepareMs)} | ` +
        `${report.maxAbsError.toExponential(2)} |`,
    );
  }
});
