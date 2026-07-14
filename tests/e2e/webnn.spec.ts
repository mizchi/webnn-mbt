import { expect, test } from "@playwright/test";

type BenchmarkReport = {
  size: number;
  warmup: number;
  iterations: number;
  batchSize: number;
  device: string;
  contextMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnFreshSamplesMs: number[];
  webnnSamplesMs: number[];
  cpuChecksum: number;
  webnnChecksum: number;
  maxAbsError: number;
};

type TransformerBenchmarkReport = {
  tokens: number;
  width: number;
  hiddenSize: number;
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  contextMs: number;
  graphMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnSamplesMs: number[];
  cpuChecksum: number;
  webnnChecksum: number;
  maxAbsError: number;
};

type AttentionBenchmarkReport = {
  tokens: number;
  width: number;
  heads: number;
  maskMode: "none" | "causal" | "padding" | "causal-padding";
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  contextMs: number;
  graphMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnSamplesMs: number[];
  maxAbsError: number;
};

type EncoderBenchmarkReport = {
  batchSize: number;
  tokens: number;
  width: number;
  hiddenSize: number;
  heads: number;
  maskMode: "none" | "causal" | "padding" | "causal-padding";
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  contextMs: number;
  graphMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnSamplesMs: number[];
  cpuChecksum: number;
  webnnChecksum: number;
  maxAbsError: number;
};

type EncoderStackBenchmarkReport = EncoderBenchmarkReport & {
  layers: number;
};

type BertEncoderBenchmarkReport = EncoderStackBenchmarkReport & {
  architecture: "bert-post-norm";
  tensorCount: number;
  loadMs: number;
};

type BertSafetensorsFixtureReport = {
  fileBytes: number;
  tensorCount: number;
  layers: number;
  inputShape: number[];
  outputShape: number[];
  cpuOutput: number[];
  webnnOutput: number[];
  maxAbsError: number;
};

type MnistEvaluationReport = {
  count: number;
  cpuCorrect: number;
  webnnCorrect: number;
  predictionMatches: number;
  maxAbsError: number;
  labels: number[];
  cpuPredictions: number[];
  webnnPredictions: number[];
  cpuLogits: number[];
  webnnLogits: number[];
};

type MnistBenchmarkReport = {
  batchSize: number;
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  fixtureMs: number;
  contextMs: number;
  graphMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnFreshSamplesMs: number[];
  webnnSamplesMs: number[];
  cpuCorrect: number;
  webnnCorrect: number;
  predictionMatches: number;
  maxAbsError: number;
};

type MnistCachedReport = {
  batchSize: number;
  cacheHit: boolean;
  cacheSize: number;
  logits: number[];
  predictions: number[];
};

type MnistCacheBenchmarkReport = {
  batchSize: number;
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  coldSamplesMs: number[];
  hitSamplesMs: number[];
  coldMisses: number;
  cacheSize: number;
  predictionMatches: number;
  maxAbsError: number;
};

type ProgramConcurrencyReport = {
  results: number[][];
  rejectedRunRecovered: boolean;
  resultAfterRejectedRun: number[];
  runAfterDestroyRejected: boolean;
};

type ExecutionPoolReport = {
  results: number[][];
  poolSize: number;
  maximumConcurrency: number;
  runAfterDestroyRejected: boolean;
};

type MnistPoolBenchmarkReport = {
  poolSize: number;
  requestCount: number;
  warmup: number;
  iterations: number;
  setupMs: number;
  samplesMs: number[];
  maximumConcurrency: number;
  predictionMatches: number;
  maxAbsError: number;
};

type NamedMultiIoReport = {
  names: string[];
  outputs: Record<string, number[]>;
  duplicateInputRejected: boolean;
  unknownInputRejected: boolean;
  inputCountRejected: boolean;
  inputLengthRejected: boolean;
  duplicateOutputRejected: boolean;
};

type NamedIoBenchmarkReport = {
  size: number;
  warmup: number;
  iterations: number;
  operationsPerSample: number;
  oneOutputSamplesMs: number[];
  twoOutputSamplesMs: number[];
  maxAbsError: number;
};

type TinyCnnBenchmarkReport = {
  inputLayout: "nchw" | "nhwc";
  batchSize: number;
  operationsPerSample: number;
  warmup: number;
  iterations: number;
  contextMs: number;
  graphMs: number;
  compileMs: number;
  prepareMs: number;
  cpuSamplesMs: number[];
  webnnSamplesMs: number[];
  predictionMatches: number;
  maxAbsError: number;
  maxProbabilitySumError: number;
};

type PreferredLayoutConvReport = {
  preferredInputLayout: "nchw" | "nhwc";
  inputShape: number[];
  filterShape: number[];
  output: number[];
};

type LiteRtBenchmarkReport = {
  warmup: number;
  iterations: number;
  operationsPerSample: number;
  contextMs: number;
  lowerMs: number;
  compileMs: number;
  prepareMs: number;
  samplesMs: number[];
  output: number[];
};

type TfliteBenchmarkReport = LiteRtBenchmarkReport & {
  parseMs: number;
};

type TfliteRunnerCacheReport = {
  first: number[];
  second: number[];
  cacheSize: number;
};

type TfliteRuntimePolicyReport = {
  hits: number;
  misses: number;
  evictions: number;
  entries: number;
  cpuFallbacks: number;
};

type TfliteRunnerCacheConcurrencyReport = {
  results: number[][];
  cacheSize: number;
};

type TfliteRunnerCacheClearRaceReport = {
  output: number[];
  cacheSizeAfterClear: number;
  residentBytesAfterClear: number;
};

type TfliteRuntimeCpuFallbackCacheReport = {
  first: number[];
  second: number[];
  cpuFallbacks: number;
  cpuCacheHits: number;
  cpuCacheMisses: number;
  cpuCacheEntries: number;
  cpuCacheResidentBytes: number;
};

type TfliteRuntimeByteBudgetReport = {
  hits: number;
  misses: number;
  evictions: number;
  entries: number;
  residentBytes: number;
  byteCapacity: number;
};

type TfliteRunnerMultiIoReport = {
  names: string[];
  outputs: Record<string, number[]>;
};

type ProgramCacheClearRaceReport = {
  output: number[];
  cacheHit: boolean;
  cacheSizeAfterClear: number;
};

type WebNNCapabilitiesReport = {
  compatibilityMode: "accelerated" | "legacy-device-type";
  preferredInputLayout: "nchw" | "nhwc";
  opSupportLimitsAvailable: boolean;
  tensorIoAvailable: boolean;
  supportedOperators: string[];
};

type MicroSpeechFixtureReport = {
  error?: string;
  inputName: string;
  inputShape: number[];
  outputName: string;
  outputShape: number[];
  first: number[];
  second: number[];
  cacheSize: number;
};

type MobileNetV2FixtureReport = {
  error?: string;
  inputShape: number[];
  outputShape: number[];
};

type UnsupportedTfliteFixtureReport = {
  error: string;
};

declare global {
  interface Window {
    webnnPlayground?: {
      compatibilityMode(): string;
      supportedOperators(): Promise<string[]>;
      probeWebnnCapabilities(): Promise<WebNNCapabilitiesReport>;
      runCpuLinear(): Promise<number[]>;
      runAdd(): Promise<number[]>;
      runConcat(): Promise<number[]>;
      runLinear(): Promise<number[]>;
      runLiteRtLinear(): Promise<number[]>;
      runTfliteAdd(): Promise<number[]>;
      runTfliteAddRelu6(): Promise<number[]>;
      runTfliteFullyConnected(): Promise<number[]>;
      runTfliteFullyConnectedRelu6(): Promise<number[]>;
      runTfliteConv2d(): Promise<number[]>;
      runTfliteReshape(): Promise<number[]>;
      runTfliteTranspose(): Promise<number[]>;
      runTfliteDepthwiseConv2d(): Promise<number[]>;
      runTfliteAveragePool2d(): Promise<number[]>;
      runTfliteMaxPool2d(): Promise<number[]>;
      runTfliteLogistic(): Promise<number[]>;
      runTfliteTanh(): Promise<number[]>;
      runTfliteQuantizedAdd(): Promise<number[]>;
      runTflitePerAxisInt8Raw(): Promise<number[]>;
      runTfliteReluN1To1(): Promise<number[]>;
      runTfliteRelu6(): Promise<number[]>;
      runTfliteQuantizedAddRaw(): Promise<number[]>;
      benchmarkLiteRtLinear(
        warmup: number,
        iterations: number,
      ): Promise<LiteRtBenchmarkReport>;
      benchmarkTfliteFullyConnected(
        warmup: number,
        iterations: number,
      ): Promise<TfliteBenchmarkReport>;
      benchmarkTfliteSub(
        warmup: number,
        iterations: number,
      ): Promise<TfliteBenchmarkReport>;
      benchmarkTfliteDepthwiseConv2d(
        warmup: number,
        iterations: number,
      ): Promise<TfliteBenchmarkReport>;
      benchmarkTfliteMaxPool2d(
        warmup: number,
        iterations: number,
      ): Promise<TfliteBenchmarkReport>;
      benchmarkTflitePerAxisInt8Raw(
        warmup: number,
        iterations: number,
      ): Promise<TfliteBenchmarkReport>;
      benchmarkTfliteRunnerCachePerAxisInt8Raw(
        warmup: number,
        iterations: number,
      ): Promise<{
        coldSamplesMs: number[];
        hitSamplesMs: number[];
        cacheSize: number;
        output: number[];
      }>;
      runTfliteRunnerCachePerAxisInt8Raw(): Promise<TfliteRunnerCacheReport>;
      runTfliteRuntimeCachePerAxisInt8Raw(): Promise<TfliteRunnerCacheReport>;
      runTfliteCpuPerAxisInt8Raw(): Promise<number[]>;
      runTfliteRuntimeCpuFallbackPerAxisInt8Raw(): Promise<{
        output: number[];
        misses: number;
        entries: number;
        cpuFallbacks: number;
      }>;
      probeTfliteRuntimePolicy(): Promise<TfliteRuntimePolicyReport>;
      probeTfliteRuntimeCpuFallbackCache(): Promise<TfliteRuntimeCpuFallbackCacheReport>;
      probeTfliteRuntimeByteBudget(): Promise<TfliteRuntimeByteBudgetReport>;
      runMicroSpeechFixture(): Promise<MicroSpeechFixtureReport>;
      runPersonDetectionFixture(): Promise<MicroSpeechFixtureReport>;
      probeMobileNetV2Fixture(): Promise<MobileNetV2FixtureReport>;
      runMobileNetV2Fixture(): Promise<MicroSpeechFixtureReport>;
      benchmarkMobileNetV2Fixture(
        warmup: number,
        iterations: number,
      ): Promise<{
        warmup: number;
        iterations: number;
        coldSamplesMs: number[];
        hitSamplesMs: number[];
        preparedHitSamplesMs: number[];
        output: number[];
        cacheSize: number;
      }>;
      probeAudioPreprocessorFixture(): Promise<UnsupportedTfliteFixtureReport>;
      probeTfliteRunnerCacheConcurrency(): Promise<TfliteRunnerCacheConcurrencyReport>;
      probeTfliteRunnerCacheClearDuringCompile(): Promise<TfliteRunnerCacheClearRaceReport>;
      runTfliteRunnerNamedAdd(): Promise<number[]>;
      runTfliteRunnerTwoInputTwoOutput(): Promise<TfliteRunnerMultiIoReport>;
      runTfliteSub(): Promise<number[]>;
      runTransformPipeline(): Promise<number[]>;
      benchmarkMatmul(
        size: number,
        warmup: number,
        iterations: number,
      ): Promise<BenchmarkReport>;
      benchmarkTransformerBlock(
        tokens: number,
        width: number,
        warmup: number,
        iterations: number,
      ): Promise<TransformerBenchmarkReport>;
      benchmarkSelfAttention(
        tokens: number,
        width: number,
        heads: number,
        maskMode: "none" | "causal" | "padding" | "causal-padding",
        warmup: number,
        iterations: number,
      ): Promise<AttentionBenchmarkReport>;
      benchmarkTransformerEncoder(
        batchSize: number,
        tokens: number,
        width: number,
        heads: number,
        maskMode: "none" | "causal" | "padding" | "causal-padding",
        warmup: number,
        iterations: number,
      ): Promise<EncoderBenchmarkReport>;
      benchmarkTransformerEncoderStack(
        layers: number,
        batchSize: number,
        tokens: number,
        width: number,
        heads: number,
        maskMode: "none" | "causal" | "padding" | "causal-padding",
        warmup: number,
        iterations: number,
      ): Promise<EncoderStackBenchmarkReport>;
      benchmarkBertEncoder(
        layers: number,
        batchSize: number,
        tokens: number,
        width: number,
        heads: number,
        maskMode: "none" | "causal" | "padding" | "causal-padding",
        warmup: number,
        iterations: number,
      ): Promise<BertEncoderBenchmarkReport>;
      runBertSafetensorsFixture(): Promise<BertSafetensorsFixtureReport>;
      evaluateMnist(count: number): Promise<MnistEvaluationReport>;
      benchmarkMnist(
        batchSize: number,
        warmup: number,
        iterations: number,
      ): Promise<MnistBenchmarkReport>;
      clearMnistCache(): number;
      runMnistCached(batchSize: number): Promise<MnistCachedReport>;
      benchmarkMnistCache(
        batchSize: number,
        warmup: number,
        iterations: number,
      ): Promise<MnistCacheBenchmarkReport>;
      probeProgramCacheClearDuringCompile(): Promise<ProgramCacheClearRaceReport>;
      probeConcurrentProgram(): Promise<ProgramConcurrencyReport>;
      probeExecutionPool(): Promise<ExecutionPoolReport>;
      benchmarkMnistPool(
        poolSize: number,
        requestCount: number,
        warmup: number,
        iterations: number,
      ): Promise<MnistPoolBenchmarkReport>;
      runNamedMultiIo(): Promise<NamedMultiIoReport>;
      benchmarkNamedIo(
        size: number,
        warmup: number,
        iterations: number,
      ): Promise<NamedIoBenchmarkReport>;
      runConvSoftmax(): Promise<number[]>;
      runGroupedConv(): Promise<number[]>;
      benchmarkTinyCnn(
        batchSize: number,
        warmup: number,
        iterations: number,
      ): Promise<TinyCnnBenchmarkReport>;
      benchmarkTinyCnnLayout(
        inputLayout: "nchw" | "nhwc",
        batchSize: number,
        warmup: number,
        iterations: number,
      ): Promise<TinyCnnBenchmarkReport>;
      runPreferredLayoutConv(): Promise<PreferredLayoutConvReport>;
    };
  }
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("MoonBit API is exposed in Canary headless", async ({ page }) => {
  await expect
    .poll(() => page.evaluate(() => typeof window.webnnPlayground?.runAdd))
    .toBe("function");
  await expect(page.evaluate(() => typeof navigator.ml)).resolves.toBe("object");
});

test("WebNN executes an addition graph", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runAdd());
  expect(result).toEqual([11, 22, 33, 44]);
});

test("WebNN concatenates named inputs along an inner axis", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runConcat());
  expect(result).toEqual([1, 2, 10, 3, 4, 20]);
});

test("generic Linear model agrees with its CPU reference", async ({ page }) => {
  const [cpu, webnn] = await page.evaluate(async () => [
    await window.webnnPlayground!.runCpuLinear(),
    await window.webnnPlayground!.runLinear(),
  ]);
  expect(cpu).toEqual([8, 11]);
  expect(webnn).toEqual(cpu);
});

test("LiteRT-compatible IR lowers a linear graph to WebNN", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runLiteRtLinear(),
  );
  expect(result).toEqual([8, 11]);
});

test("TFLite FlatBuffer parses and lowers to WebNN", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runTfliteAdd());
  expect(result).toEqual([8, 11]);
});

test("TFLite ADD fused RELU6 lowers to WebNN clamp", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteAddRelu6(),
  );
  expect(result).toEqual([6, 6]);
});

test("TFLite FULLY_CONNECTED lowers weight layout and bias to WebNN", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteFullyConnected(),
  );
  expect(result).toEqual([8, 11]);
});

test("TFLite FULLY_CONNECTED fused RELU6 runs after bias add", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteFullyConnectedRelu6(),
  );
  expect(result).toEqual([6, 6]);
});

test("TFLite CONV_2D lowers NHWC/OHWI and bias to WebNN", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteConv2d(),
  );
  expect(result).toEqual([-4, -4, -4, -4]);
});

test("TFLite RESHAPE lowers an int32 shape constant to WebNN", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteReshape(),
  );
  expect(result).toEqual([1, 2, 3, 4]);
});

test("TFLite TRANSPOSE lowers an int32 permutation constant to WebNN", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteTranspose(),
  );
  expect(result).toEqual([1, 4, 2, 5, 3, 6]);
});

test("TFLite DEPTHWISE_CONV_2D lowers to grouped NHWC/HWIO WebNN conv2d", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteDepthwiseConv2d(),
  );
  expect(result).toEqual([63, 730, 79, 930, 111, 1330, 127, 1530]);
});

test("TFLite AVERAGE_POOL_2D lowers to WebNN", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteAveragePool2d(),
  );
  expect(result).toEqual([3, 4, 6, 7]);
});

test("TFLite MAX_POOL_2D lowers to WebNN", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteMaxPool2d(),
  );
  expect(result).toEqual([5, 6, 8, 9]);
});

test("TFLite LOGISTIC lowers to WebNN sigmoid", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteLogistic(),
  );
  expect(result[0]).toBeCloseTo(0.5, 6);
  expect(result[1]).toBeCloseTo(0.7310586, 6);
});

test("TFLite TANH lowers to WebNN tanh", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runTfliteTanh());
  expect(result[0]).toBeCloseTo(0, 6);
  expect(result[1]).toBeCloseTo(0.7615942, 6);
});

test("quantized UINT8 TFLite constants dequantize before WebNN lowering", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteQuantizedAdd(),
  );
  expect(result).toEqual([2, 4]);
});

test("TfliteRunner preserves raw per-axis INT8 I/O and reuses a cached program", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteRunnerCachePerAxisInt8Raw(),
  );
  expect(result.first).toEqual([5, 10]);
  expect(result.second).toEqual([5, 10]);
  expect(result.cacheSize).toBe(1);
});

test("WebNNRuntime owns a reusable TFLite program cache", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteRuntimeCachePerAxisInt8Raw(),
  );
  expect(result).toEqual({
    first: [5, 10],
    second: [5, 10],
    cacheSize: 1,
  });
});

test("TfliteCpuRunner preserves raw per-axis INT8 I/O", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteCpuPerAxisInt8Raw(),
  );
  expect(result).toEqual([5, 10]);
});

test("WebNNRuntime falls back to CPU when navigator.ml is unavailable", async ({
  page,
}) => {
  const result = await page.evaluate(async () => {
    Object.defineProperty(navigator, "ml", {
      configurable: true,
      value: undefined,
    });
    try {
      return await window.webnnPlayground!
        .runTfliteRuntimeCpuFallbackPerAxisInt8Raw();
    } finally {
      delete (navigator as Navigator & { ml?: unknown }).ml;
    }
  });
  expect(result).toEqual({
    output: [5, 10],
    misses: 1,
    entries: 0,
    cpuFallbacks: 1,
  });
});

test("WebNNRuntime reuses a parsed CPU model across fallback calls", async ({
  page,
}) => {
  const result = await page.evaluate(async () => {
    Object.defineProperty(navigator, "ml", {
      configurable: true,
      value: undefined,
    });
    try {
      return await window.webnnPlayground!.probeTfliteRuntimeCpuFallbackCache();
    } finally {
      delete (navigator as Navigator & { ml?: unknown }).ml;
    }
  });
  expect(result.first).toEqual([5, 10]);
  expect(result.second).toEqual([5, 10]);
  expect(result.cpuFallbacks).toBe(2);
  expect(result.cpuCacheHits).toBe(1);
  expect(result.cpuCacheMisses).toBe(1);
  expect(result.cpuCacheEntries).toBe(1);
  expect(result.cpuCacheResidentBytes).toBeGreaterThan(0);
});

test("WebNNRuntime bounds its TFLite cache with LRU metrics", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.probeTfliteRuntimePolicy(),
  );
  expect(result).toEqual({
    hits: 1,
    misses: 4,
    evictions: 2,
    entries: 2,
    cpuFallbacks: 0,
  });
});

test("WebNNRuntime evicts TFLite runners by source-byte budget", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.probeTfliteRuntimeByteBudget(),
  );
  expect(result).toMatchObject({
    hits: 0,
    misses: 3,
    evictions: 2,
    entries: 1,
  });
  expect(result.residentBytes).toBeGreaterThan(0);
  expect(result.residentBytes).toBeLessThanOrEqual(result.byteCapacity);
});

test("a real Micro Speech TFLite fixture lowers and reuses WebNNRuntime", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runMicroSpeechFixture(),
  );
  expect(result.error).toBeUndefined();
  expect(result.inputShape).toEqual([1, 1960]);
  expect(result.outputShape).toEqual([1, 4]);
  expect(result.first).toHaveLength(4);
  expect(result.first).toEqual(result.second);
  expect(result.first.every((value) => value >= -128 && value <= 127)).toBe(
    true,
  );
  expect(result.cacheSize).toBe(1);
});

test("a real person-detection TFLite fixture lowers and reuses WebNNRuntime", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runPersonDetectionFixture(),
  );
  expect(result.error).toBeUndefined();
  expect(result.inputShape).toEqual([1, 96, 96, 1]);
  expect(result.outputShape).toEqual([1, 2]);
  expect(result.first).toHaveLength(2);
  expect(result.first).toEqual(result.second);
  expect(result.first.every((value) => value >= -128 && value <= 127)).toBe(
    true,
  );
  expect(result.cacheSize).toBe(1);
});

test("a real MobileNet V2 fixture parses through the supported TFLite subset", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.probeMobileNetV2Fixture(),
  );
  expect(result.error).toBeUndefined();
  expect(result.inputShape).toEqual([1, 224, 224, 3]);
  expect(result.outputShape).toHaveLength(2);
});

test("a real MobileNet V2 fixture lowers and reuses WebNNRuntime", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runMobileNetV2Fixture(),
  );
  expect(result.error).toBeUndefined();
  expect(result.inputShape).toEqual([1, 224, 224, 3]);
  expect(result.first).toHaveLength(result.outputShape[1]);
  expect(result.first).toEqual(result.second);
  expect(result.first.every((value) => value >= 0 && value <= 255)).toBe(true);
  expect(result.cacheSize).toBe(1);
});

test("MobileNet V2 benchmark separates cache miss from cache hit", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkMobileNetV2Fixture(0, 1),
  );
  expect(result).toMatchObject({
    warmup: 0,
    iterations: 1,
    cacheSize: 1,
  });
  expect(result.coldSamplesMs).toHaveLength(1);
  expect(result.hitSamplesMs).toHaveLength(1);
  expect(result.preparedHitSamplesMs).toHaveLength(1);
  expect(result.output.length).toBeGreaterThan(0);
});

test("a real Audio Preprocessor fixture names its unsupported custom operator", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.probeAudioPreprocessorFixture(),
  );
  expect(result).toEqual({
    error: "unsupported TFLite custom operator: SignalWindow",
  });
});

test("TfliteRunnerCache keeps concurrent raw INT8 runs isolated", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.probeTfliteRunnerCacheConcurrency(),
  );
  expect(result.results).toEqual([
    [5, 10],
    [6, 10],
  ]);
  expect(result.cacheSize).toBe(1);
});

test("TfliteRunner uses declared TFLite tensor names for named I/O", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteRunnerNamedAdd(),
  );
  expect(result).toEqual([8, 11]);
});

test("TfliteRunner binds multiple TFLite inputs by name and preserves output order", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteRunnerTwoInputTwoOutput(),
  );
  expect(result.names).toEqual(["tensor_2", "tensor_3"]);
  expect(result.outputs).toEqual({
    tensor_2: [11, 22],
    tensor_3: [1, 2],
  });
});

test("TFLite SUB lowers through the WebNN sub operator", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteSub(),
  );
  expect(result).toEqual([2, 10]);
});

test("TFLite RELU_N1_TO_1 lowers to WebNN clamp", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runTfliteReluN1To1());
  expect(result).toEqual([-1, -0.5, 0.5, 1]);
});

test("TFLite RELU6 lowers to WebNN clamp", async ({ page }) => {
  const result = await page.evaluate(() => window.webnnPlayground!.runTfliteRelu6());
  expect(result).toEqual([0, 0.5, 5, 6]);
});

test("raw UINT8 input and output values round-trip through a float32 WebNN graph", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTfliteQuantizedAddRaw(),
  );
  expect(result).toEqual([132, 136]);
});

test("per-axis INT8 input and output values round-trip through a TFLite WebNN graph", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTflitePerAxisInt8Raw(),
  );
  expect(result).toEqual([5, 10]);
});

test("TFLite benchmark separates FlatBuffer parsing from lowering", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTfliteFullyConnected(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [8, 11],
  });
  expect(report.parseMs).toBeGreaterThanOrEqual(0);
  expect(report.lowerMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
});

test("TFLite SUB benchmark measures named two-input execution", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTfliteSub(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [2, 10],
  });
  expect(report.parseMs).toBeGreaterThanOrEqual(0);
  expect(report.lowerMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
});

test("TFLite depthwise benchmark includes parse, lower, and repeated WebNN execution", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTfliteDepthwiseConv2d(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [63, 730, 79, 930, 111, 1330, 127, 1530],
  });
  expect(report.parseMs).toBeGreaterThanOrEqual(0);
  expect(report.lowerMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
});

test("TFLite max-pool benchmark includes parse, lower, and repeated WebNN execution", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTfliteMaxPool2d(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [5, 6, 8, 9],
  });
  expect(report.parseMs).toBeGreaterThanOrEqual(0);
  expect(report.lowerMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
});

test("per-axis INT8 TFLite benchmark includes raw conversion and WebNN execution", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTflitePerAxisInt8Raw(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [2, 3],
  });
  expect(report.parseMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
});

test("LiteRT lowering benchmark separates lowering from repeated execution", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkLiteRtLinear(1, 3),
  );
  expect(report).toMatchObject({
    warmup: 1,
    iterations: 3,
    output: [8, 11],
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.samplesMs).toHaveLength(3);
  expect(report.contextMs).toBeGreaterThanOrEqual(0);
  expect(report.lowerMs).toBeGreaterThanOrEqual(0);
  expect(report.compileMs).toBeGreaterThanOrEqual(0);
  expect(report.prepareMs).toBeGreaterThanOrEqual(0);
});

test("WebNN composes gather, arithmetic, normalization, gelu, reshape, and transpose", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runTransformPipeline(),
  );
  expect(result[0]).toBeCloseTo(0.1497, 3);
});

test("compatibility adapter detects the Canary WebNN contract", async ({ page }) => {
  const mode = await page.evaluate(() =>
    window.webnnPlayground!.compatibilityMode(),
  );
  expect(["accelerated", "legacy-device-type"]).toContain(mode);
});

test("backend validates the MVP operators against opSupportLimits", async ({
  page,
}) => {
  const operators = await page.evaluate(() =>
    window.webnnPlayground!.supportedOperators(),
  );
  expect(operators).toEqual(
    expect.arrayContaining([
      "add",
      "sub",
      "mul",
      "div",
      "reduceMean",
      "concat",
      "matmul",
      "conv2d",
      "maxPool2d",
      "averagePool2d",
      "sigmoid",
      "tanh",
      "gelu",
      "layerNormalization",
      "clamp",
      "relu",
      "softmax",
      "reshape",
      "transpose",
    ]),
  );
});

test("Canary exposes the WebNN capability contract used by this backend", async ({
  page,
}) => {
  const capabilities = await page.evaluate(() =>
    window.webnnPlayground!.probeWebnnCapabilities(),
  );
  expect(["accelerated", "legacy-device-type"]).toContain(
    capabilities.compatibilityMode,
  );
  expect(capabilities.preferredInputLayout).toMatch(/^(nchw|nhwc)$/);
  expect(capabilities.opSupportLimitsAvailable).toBe(true);
  expect(capabilities.tensorIoAvailable).toBe(true);
  expect(capabilities.supportedOperators).toEqual(
    expect.arrayContaining([
      "add",
      "sub",
      "mul",
      "div",
      "reduceMean",
      "concat",
      "matmul",
      "conv2d",
      "maxPool2d",
      "averagePool2d",
      "sigmoid",
      "tanh",
      "gelu",
      "layerNormalization",
      "clamp",
      "relu",
      "softmax",
      "reshape",
      "transpose",
    ]),
  );
});

test("benchmark separates setup and repeated inference timings", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkMatmul(8, 1, 2),
  );
  expect(report).toMatchObject({
    size: 8,
    warmup: 1,
    iterations: 2,
    device: "npu",
  });
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.batchSize).toBeGreaterThan(0);
  expect(report.webnnFreshSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.contextMs).toBeGreaterThanOrEqual(0);
  expect(report.compileMs).toBeGreaterThanOrEqual(0);
  expect(report.prepareMs).toBeGreaterThanOrEqual(0);
  expect(report.maxAbsError).toBeLessThan(1e-4);
  expect(report.cpuChecksum).toBeCloseTo(report.webnnChecksum, 4);
});

test("Transformer block benchmark agrees between CPU and WebNN", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTransformerBlock(2, 4, 1, 2),
  );
  expect(report).toMatchObject({
    tokens: 2,
    width: 4,
    hiddenSize: 8,
    warmup: 1,
    iterations: 2,
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.contextMs).toBeGreaterThanOrEqual(0);
  expect(report.graphMs).toBeGreaterThanOrEqual(0);
  expect(report.compileMs).toBeGreaterThanOrEqual(0);
  expect(report.prepareMs).toBeGreaterThanOrEqual(0);
  expect(report.maxAbsError).toBeLessThan(1e-3);
  expect(report.cpuChecksum).toBeCloseTo(report.webnnChecksum, 3);
});

test("WebNN self-attention supports heads and additive masks", async ({
  page,
}) => {
  const reports = await page.evaluate(async () => {
    const benchmark = window.webnnPlayground!.benchmarkSelfAttention;
    return [
      await benchmark(4, 4, 1, "none", 1, 2),
      await benchmark(4, 4, 2, "causal", 1, 2),
      await benchmark(4, 4, 2, "padding", 1, 2),
      await benchmark(4, 4, 2, "causal-padding", 1, 2),
    ];
  });
  expect(reports.map((report) => report.maskMode)).toEqual([
    "none",
    "causal",
    "padding",
    "causal-padding",
  ]);
  expect(reports.map((report) => report.heads)).toEqual([1, 2, 2, 2]);
  for (const report of reports) {
    expect(report).toMatchObject({
      tokens: 4,
      width: 4,
      warmup: 1,
      iterations: 2,
    });
    expect(report.cpuSamplesMs).toHaveLength(2);
    expect(report.webnnSamplesMs).toHaveLength(2);
    expect(report.maxAbsError).toBeLessThan(1e-3);
  }
});

test("WebNN Transformer encoder composes batched attention and feed-forward", async ({
  page,
}) => {
  const [report, largeReport] = await page.evaluate(async () => [
    await window.webnnPlayground!.benchmarkTransformerEncoder(
      2,
      4,
      4,
      2,
      "causal-padding",
      1,
      2,
    ),
    await window.webnnPlayground!.benchmarkTransformerEncoder(
      2,
      16,
      256,
      4,
      "causal-padding",
      0,
      1,
    ),
  ]);
  expect(report).toMatchObject({
    batchSize: 2,
    tokens: 4,
    width: 4,
    hiddenSize: 8,
    heads: 2,
    maskMode: "causal-padding",
    warmup: 1,
    iterations: 2,
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.maxAbsError).toBeLessThan(1e-3);
  expect(report.cpuChecksum).toBeCloseTo(report.webnnChecksum, 3);
  expect(largeReport).toMatchObject({
    batchSize: 2,
    tokens: 16,
    width: 256,
    hiddenSize: 512,
    heads: 4,
    maskMode: "causal-padding",
  });
  expect(largeReport.maxAbsError).toBeLessThan(3e-3);
});

test("WebNN Transformer encoder stack compiles distinct layers into one graph", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTransformerEncoderStack(
      2,
      2,
      4,
      4,
      2,
      "causal-padding",
      1,
      2,
    ),
  );
  expect(report).toMatchObject({
    layers: 2,
    batchSize: 2,
    tokens: 4,
    width: 4,
    hiddenSize: 8,
    heads: 2,
    maskMode: "causal-padding",
    warmup: 1,
    iterations: 2,
  });
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.maxAbsError).toBeLessThan(3e-3);
});

test("Hugging Face named tensors load into a post-norm BERT WebNN graph", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkBertEncoder(
      2,
      2,
      4,
      4,
      2,
      "padding",
      1,
      2,
    ),
  );
  expect(report).toMatchObject({
    architecture: "bert-post-norm",
    layers: 2,
    tensorCount: 32,
    batchSize: 2,
    tokens: 4,
    width: 4,
    hiddenSize: 8,
    heads: 2,
    maskMode: "padding",
    warmup: 1,
    iterations: 2,
  });
  expect(report.loadMs).toBeGreaterThanOrEqual(0);
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.maxAbsError).toBeLessThan(5e-3);
});

test("a SafeTensors BERT fixture parses, maps, and executes through WebNN", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.runBertSafetensorsFixture(),
  );
  expect(report).toMatchObject({
    fileBytes: 8288,
    tensorCount: 32,
    layers: 2,
    inputShape: [2, 4, 8],
    outputShape: [2, 4, 8],
  });
  expect(report.cpuOutput).toHaveLength(64);
  expect(report.webnnOutput).toHaveLength(64);
  expect(report.maxAbsError).toBeLessThan(5e-5);
});

test("Tiny CNN runs with NHWC/HWIO while retaining CPU agreement", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTinyCnnLayout("nhwc", 2, 1, 3),
  );
  expect(report).toMatchObject({
    inputLayout: "nhwc",
    batchSize: 2,
    warmup: 1,
    iterations: 3,
    predictionMatches: 2,
  });
  expect(report.cpuSamplesMs).toHaveLength(3);
  expect(report.webnnSamplesMs).toHaveLength(3);
  expect(report.maxAbsError).toBeLessThan(1e-4);
  expect(report.maxProbabilitySumError).toBeLessThan(1e-5);
});

test("MNIST MLP matches golden logits and CPU predictions", async ({ page }) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.evaluateMnist(100),
  );
  expect(report).toMatchObject({
    count: 100,
    cpuCorrect: 96,
    webnnCorrect: 96,
    predictionMatches: 100,
  });
  expect(report.labels[0]).toBe(7);
  expect(report.cpuPredictions[0]).toBe(7);
  expect(report.webnnPredictions[0]).toBe(7);
  expect(report.maxAbsError).toBeLessThan(1e-4);

  const golden = [
    -0.93861, -7.460587, 1.949085, 4.563724, -4.459607,
    -0.70809, -7.971023, 9.501954, -3.350631, 0.21702,
  ];
  for (const [index, expected] of golden.entries()) {
    expect(report.cpuLogits[index]).toBeCloseTo(expected, 4);
    expect(report.webnnLogits[index]).toBeCloseTo(expected, 4);
  }
});

test("MNIST benchmark separates model setup and inference", async ({ page }) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkMnist(1, 1, 2),
  );
  expect(report).toMatchObject({
    batchSize: 1,
    warmup: 1,
    iterations: 2,
    cpuCorrect: 1,
    webnnCorrect: 1,
    predictionMatches: 1,
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.cpuSamplesMs).toHaveLength(2);
  expect(report.webnnFreshSamplesMs).toHaveLength(2);
  expect(report.webnnSamplesMs).toHaveLength(2);
  expect(report.fixtureMs).toBeGreaterThanOrEqual(0);
  expect(report.graphMs).toBeGreaterThanOrEqual(0);
  expect(report.compileMs).toBeGreaterThanOrEqual(0);
  expect(report.maxAbsError).toBeLessThan(1e-4);
});

test("MNIST compiled program cache misses once then reuses the execution", async ({
  page,
}) => {
  const result = await page.evaluate(async () => {
    window.webnnPlayground!.clearMnistCache();
    const first = await window.webnnPlayground!.runMnistCached(16);
    const second = await window.webnnPlayground!.runMnistCached(16);
    const differentShape = await window.webnnPlayground!.runMnistCached(1);
    return { first, second, differentShape };
  });

  expect(result.first).toMatchObject({
    batchSize: 16,
    cacheHit: false,
    cacheSize: 1,
  });
  expect(result.second).toMatchObject({
    batchSize: 16,
    cacheHit: true,
    cacheSize: 1,
  });
  expect(result.second.logits).toEqual(result.first.logits);
  expect(result.second.predictions).toEqual(result.first.predictions);
  expect(result.differentShape).toMatchObject({
    batchSize: 1,
    cacheHit: false,
    cacheSize: 2,
  });
});

test("MNIST cache benchmark separates cold miss and cache-hit inference", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkMnistCache(16, 1, 3),
  );
  expect(report).toMatchObject({
    batchSize: 16,
    operationsPerSample: 1,
    warmup: 1,
    iterations: 3,
    coldMisses: 3,
    cacheSize: 1,
    predictionMatches: 16,
  });
  expect(report.coldSamplesMs).toHaveLength(3);
  expect(report.coldSamplesMs.every((sample) => sample >= 0)).toBe(true);
  expect(report.hitSamplesMs).toHaveLength(3);
  expect(report.maxAbsError).toBeLessThan(1e-4);
});

test("program cache does not repopulate after clear during compilation", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.probeProgramCacheClearDuringCompile(),
  );
  expect(report).toEqual({
    output: [8, 11],
    cacheHit: false,
    cacheSizeAfterClear: 0,
  });
});

test("TfliteRunnerCache does not repopulate after clear during compilation", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.probeTfliteRunnerCacheClearDuringCompile(),
  );
  expect(report).toEqual({
    output: [8, 11],
    cacheSizeAfterClear: 0,
    residentBytesAfterClear: 0,
  });
});

test("WebNNProgram serializes concurrent runs and defers destroy", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.probeConcurrentProgram(),
  );
  expect(report.results).toEqual([
    [8, 11],
    [16, 23],
  ]);
  expect(report.rejectedRunRecovered).toBe(true);
  expect(report.resultAfterRejectedRun).toEqual([8, 11]);
  expect(report.runAfterDestroyRejected).toBe(true);
});

test("WebNNProgram execution pool assigns concurrent runs to separate slots", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.probeExecutionPool(),
  );
  expect(report.results).toEqual([
    [8, 11],
    [16, 23],
  ]);
  expect(report.poolSize).toBe(2);
  expect(report.maximumConcurrency).toBe(2);
  expect(report.runAfterDestroyRejected).toBe(true);
});

test("MNIST execution pool benchmark measures concurrent request groups", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkMnistPool(2, 4, 1, 3),
  );
  expect(report).toMatchObject({
    poolSize: 2,
    requestCount: 4,
    warmup: 1,
    iterations: 3,
    maximumConcurrency: 2,
    predictionMatches: 4,
  });
  expect(report.setupMs).toBeGreaterThanOrEqual(0);
  expect(report.samplesMs).toHaveLength(3);
  expect(report.maxAbsError).toBeLessThan(1e-4);
});

test("WebNNProgram binds multiple inputs and outputs by name", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.runNamedMultiIo(),
  );
  expect(report.names).toEqual(["product", "sum"]);
  expect(report.outputs).toEqual({
    product: [3, 8],
    sum: [4, 6],
  });
  expect(report.duplicateInputRejected).toBe(true);
  expect(report.unknownInputRejected).toBe(true);
  expect(report.inputCountRejected).toBe(true);
  expect(report.inputLengthRejected).toBe(true);
  expect(report.duplicateOutputRejected).toBe(true);
});

test("named I/O benchmark compares one and two output readbacks", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkNamedIo(64, 1, 3),
  );
  expect(report).toMatchObject({
    size: 64,
    warmup: 1,
    iterations: 3,
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.oneOutputSamplesMs).toHaveLength(3);
  expect(report.twoOutputSamplesMs).toHaveLength(3);
  expect(report.maxAbsError).toBeLessThan(1e-6);
});

test("WebNN executes conv2d followed by axis softmax", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runConvSoftmax(),
  );
  expect(result).toHaveLength(4);
  expect(result[0]).toBeCloseTo(0.2689414, 5);
  expect(result[1]).toBeCloseTo(0.7310586, 5);
  expect(result[2]).toBeCloseTo(0.2689414, 5);
  expect(result[3]).toBeCloseTo(0.7310586, 5);
});

test("WebNN executes grouped NCHW/OIHW conv2d", async ({ page }) => {
  const result = await page.evaluate(() =>
    window.webnnPlayground!.runGroupedConv(),
  );
  expect(result).toEqual([
    2, 4, 6, 8,
    15, 18, 21, 24,
  ]);
});

test("WebNN executes conv2d using the context preferred input layout", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.runPreferredLayoutConv(),
  );
  expect(["nchw", "nhwc"]).toContain(report.preferredInputLayout);
  expect(report.output).toEqual([-4, -4, -4, -4]);
  if (report.preferredInputLayout === "nhwc") {
    expect(report.inputShape).toEqual([1, 3, 3, 1]);
    expect(report.filterShape).toEqual([2, 2, 1, 1]);
  } else {
    expect(report.inputShape).toEqual([1, 1, 3, 3]);
    expect(report.filterShape).toEqual([1, 1, 2, 2]);
  }
});

test("Tiny CNN benchmark matches CPU probabilities and predictions", async ({
  page,
}) => {
  const report = await page.evaluate(() =>
    window.webnnPlayground!.benchmarkTinyCnn(4, 1, 3),
  );
  expect(report).toMatchObject({
    inputLayout: "nchw",
    batchSize: 4,
    warmup: 1,
    iterations: 3,
    predictionMatches: 4,
  });
  expect(report.operationsPerSample).toBeGreaterThan(0);
  expect(report.cpuSamplesMs).toHaveLength(3);
  expect(report.webnnSamplesMs).toHaveLength(3);
  expect(report.maxAbsError).toBeLessThan(1e-4);
  expect(report.maxProbabilitySumError).toBeLessThan(1e-5);
});
