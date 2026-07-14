import { fixed, percentile, withCanaryPage } from "./canary-harness.mjs";

const port = Number(process.env.BENCH_PORT ?? 4181);
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
    () => typeof globalThis.webnnPlayground?.benchmarkLiteRtLinear === "function",
  );
  const report = await page.evaluate(
    ([warmupCount, iterationCount]) =>
      globalThis.webnnPlayground.benchmarkLiteRtLinear(
        warmupCount,
        iterationCount,
      ),
    [warmup, iterations],
  );

  console.log(
    `\nChrome ${browserVersion} Canary headless / LiteRT IR → WebNN linear / ` +
      `WebNN npu / warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| ops/sample | context ms | lower ms | compile ms | prepare ms | run p50 ms | run p95 ms | output |",
  );
  console.log("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
  console.log(
    `| ${report.operationsPerSample} | ${fixed(report.contextMs)} | ` +
      `${fixed(report.lowerMs)} | ${fixed(report.compileMs)} | ` +
      `${fixed(report.prepareMs)} | ` +
      `${fixed(percentile(report.samplesMs, 0.5))} | ` +
      `${fixed(percentile(report.samplesMs, 0.95))} | ` +
      `[${report.output.join(", ")}] |`,
  );
});
