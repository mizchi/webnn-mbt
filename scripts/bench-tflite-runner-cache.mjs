import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4182);
const warmup = Number(process.env.BENCH_WARMUP ?? 10);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 30);

if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () =>
      typeof globalThis.webnnPlayground
        ?.benchmarkTfliteRunnerCachePerAxisInt8Raw === "function",
  );
  const report = await page.evaluate(
    ([warmupCount, iterationCount]) =>
      globalThis.webnnPlayground.benchmarkTfliteRunnerCachePerAxisInt8Raw(
        warmupCount,
        iterationCount,
      ),
    [warmup, iterations],
  );

  console.log(
    "\nChrome " +
      browserVersion +
      " Canary headless / TfliteRunner cache / WebNN npu / warmup=" +
      warmup +
      " iterations=" +
      iterations,
  );
  console.log(
    "| model | cold p50 ms | cold p95 ms | hit p50 ms | hit p95 ms | cold/hit | cache entries | raw output |",
  );
  console.log(
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  );
  const coldP50 = percentile(report.coldSamplesMs, 0.5);
  const coldP95 = percentile(report.coldSamplesMs, 0.95);
  const hitP50 = percentile(report.hitSamplesMs, 0.5);
  const hitP95 = percentile(report.hitSamplesMs, 0.95);
  console.log(
    "| INT8_PER_AXIS_RAW | " +
      fixed(coldP50) +
      " | " +
      fixed(coldP95) +
      " | " +
      fixed(hitP50) +
      " | " +
      fixed(hitP95) +
      " | " +
      fixed(coldP50 / hitP50) +
      "x | " +
      report.cacheSize +
      " | [" +
      report.output.join(", ") +
      "] |",
  );
});
