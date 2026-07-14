import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4182);
const warmup = Number(process.env.BENCH_WARMUP ?? 3);
const iterations = Number(process.env.BENCH_ITERATIONS ?? 10);

if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error("BENCH_WARMUP must be a non-negative integer");
}
if (!Number.isInteger(iterations) || iterations <= 0) {
  throw new Error("BENCH_ITERATIONS must be a positive integer");
}

await withCanaryPage({ port }, async ({ page, browserVersion }) => {
  await page.waitForFunction(
    () =>
      typeof globalThis.webnnPlayground?.benchmarkMobileNetV2Fixture ===
      "function",
  );
  const report = await page.evaluate(
    ([warmupCount, iterationCount]) =>
      globalThis.webnnPlayground.benchmarkMobileNetV2Fixture(
        warmupCount,
        iterationCount,
      ),
    [warmup, iterations],
  );

  const coldP50 = percentile(report.coldSamplesMs, 0.5);
  const coldP95 = percentile(report.coldSamplesMs, 0.95);
  const hitP50 = percentile(report.hitSamplesMs, 0.5);
  const hitP95 = percentile(report.hitSamplesMs, 0.95);
  const preparedHitP50 = percentile(report.preparedHitSamplesMs, 0.5);
  const preparedHitP95 = percentile(report.preparedHitSamplesMs, 0.95);
  console.log(
    `\nChrome ${browserVersion} Canary headless / MobileNet V2 TFLite → WebNN / ` +
      `WebNN npu / warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| model | cold p50 ms | bytes hit p50/p95 ms | prepared hit p50/p95 ms | cold/prepared | cache entries | output values |",
  );
  console.log(
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  console.log(
    "| MobileNetV2 UINT8 | " +
      fixed(coldP50) +
      " | " +
      fixed(hitP50) +
      " / " +
      fixed(hitP95) +
      " | " +
      fixed(preparedHitP50) +
      " / " +
      fixed(preparedHitP95) +
      " | " +
      fixed(coldP50 / preparedHitP50) +
      "x | " +
      report.cacheSize +
      " | " +
      report.output.length +
      " |",
  );
});
