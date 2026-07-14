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
      typeof globalThis.webnnPlayground?.benchmarkTfliteFullyConnected ===
        "function" &&
      typeof globalThis.webnnPlayground?.benchmarkTfliteSub === "function" &&
      typeof globalThis.webnnPlayground?.benchmarkTfliteDepthwiseConv2d ===
        "function" &&
      typeof globalThis.webnnPlayground?.benchmarkTfliteMaxPool2d ===
        "function" &&
      typeof globalThis.webnnPlayground?.benchmarkTflitePerAxisInt8Raw ===
        "function",
  );
  const reports = await page.evaluate(
    async ([warmupCount, iterationCount]) => [
      [
        "FULLY_CONNECTED",
        await globalThis.webnnPlayground.benchmarkTfliteFullyConnected(
          warmupCount,
          iterationCount,
        ),
      ],
      [
        "SUB",
        await globalThis.webnnPlayground.benchmarkTfliteSub(
          warmupCount,
          iterationCount,
        ),
      ],
      [
        "DEPTHWISE_CONV_2D",
        await globalThis.webnnPlayground.benchmarkTfliteDepthwiseConv2d(
          warmupCount,
          iterationCount,
        ),
      ],
      [
        "MAX_POOL_2D",
        await globalThis.webnnPlayground.benchmarkTfliteMaxPool2d(
          warmupCount,
          iterationCount,
        ),
      ],
      [
        "INT8_PER_AXIS_RAW",
        await globalThis.webnnPlayground.benchmarkTflitePerAxisInt8Raw(
          warmupCount,
          iterationCount,
        ),
      ],
    ],
    [warmup, iterations],
  );

  console.log(
    `\nChrome ${browserVersion} Canary headless / TFLite FlatBuffer → WebNN / ` +
      `WebNN npu / warmup=${warmup} iterations=${iterations}`,
  );
  console.log(
    "| model | ops/sample | parse ms | context ms | lower ms | compile ms | prepare ms | run p50 ms | run p95 ms | output |",
  );
  console.log(
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  );
  for (const [model, report] of reports) {
    console.log(
      `| ${model} | ${report.operationsPerSample} | ${fixed(report.parseMs)} | ` +
        `${fixed(report.contextMs)} | ${fixed(report.lowerMs)} | ` +
        `${fixed(report.compileMs)} | ${fixed(report.prepareMs)} | ` +
        `${fixed(percentile(report.samplesMs, 0.5))} | ` +
        `${fixed(percentile(report.samplesMs, 0.95))} | ` +
        `[${report.output.join(", ")}] |`,
    );
  }
});
