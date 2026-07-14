# webnn-mbt

English | [日本語](./README.ja.md)

A WebNN backend and TFLite inference runtime for MoonBit. It provides low-level WebNN bindings, a compatibility adapter for Chrome API revisions, a typed graph builder, a program cache, execution pools, and TFLite-to-WebNN lowering. Compatibility with the browser implementation is continuously tested in headless Chrome Canary.

The current scope is `float32`, fixed-shape inference with named multi-input/output graphs. Supported operations are:

- `add` / `sub` / `mul` / `div`
- `reduce_mean` / `gather` / `slice` / `concat`
- `matmul` (rank 2 or higher, with batch-dimension broadcasting)
- `conv2d` (NCHW/OIHW and NHWC/HWIO)
- `max_pool2d` / `average_pool2d`
- `sigmoid` / `tanh` / `gelu` (the exact erf definition from the WebNN specification)
- `layer_normalization`
- `clamp` / `relu`
- `softmax`
- `reshape`
- `transpose`

## Quick start

Add the package to a MoonBit project:

```bash
moon add mizchi/webnn
```

Import the root facade from your JavaScript-target package's `moon.pkg`:

```moonbit
import {
  "mizchi/webnn",
}

supported_targets = "js"
```

The following graph adds a constant bias to a `float32` input. The returned
`WebNNProgram` owns the graph and its prepared tensors, so destroy the program
when inference is complete:

```moonbit
pub async fn add_bias() -> Array[Float] {
  let graph = @webnn.WebNNGraphBuilder::new(
    @webnn.DevicePreference::Npu,
  )
  let input = graph.input("input", @webnn.Shape::new([1, 2]))
  let bias = graph.constant(@webnn.Shape::new([1, 2]), [10.0, 20.0])
  let output = input.tensor().add(bias)
  let program = graph.compile_program_single(input, "output", output)
  defer program.destroy()
  program.run([1.0, 2.0])
}
```

Calling `add_bias()` in a WebNN-enabled browser returns `[11.0, 22.0]`.
WebNN context creation, graph compilation, dispatch, and tensor reads are
asynchronous, so the calling code must run in an `async` context.

## Requirements

- MoonBit 0.10.3 or later
- Node.js 24 or later
- pnpm
- just
- Google Chrome Canary

On macOS, Playwright uses the `chrome-canary` channel. Tests run with an isolated browser profile, so they do not depend on settings in the regular Canary `chrome://flags` page. The required feature is passed as a launch argument:

```text
--enable-features=WebMachineLearningNeuralNetwork
```

## Running

```bash
just install
just check
```

Individual tasks are also available:

```bash
just unit
just build
just e2e
just fixture-bert
just bench
just bench-mnist
just bench-mnist-cache
just bench-mnist-pool
just bench-named-io
just bench-tiny-cnn
just bench-transformer
just bench-attention
just bench-encoder
just bench-encoder-stack
just bench-bert-encoder
just bench-litert
just bench-tflite
just bench-tflite-runner-cache
just bench-mobilenet-v2
```

## Workspace

[`moon.work`](./moon.work) separates the published library from its browser consumer:

- `.` is the `mizchi/webnn` library module. Its packages live under `src/`.
- [`examples/playground`](./examples/playground) is the repository-local `mizchi/webnn-examples` application module. It depends on the workspace copy of `mizchi/webnn`, owns benchmark and MNIST fixture code, and exposes the browser API used by Playwright.

Applications can import the curated root facade instead of depending directly on implementation packages:

```moonbit
import {
  "mizchi/webnn",
}
```

The root package re-exports the validated shape, LiteRT/TFLite, WebNN graph, program, cache, and runtime contracts. Specialized model-building packages remain available under `mizchi/webnn/model`, `mizchi/webnn/bert`, and `mizchi/webnn/backend/cpu`.

`just build` builds the playground application explicitly. MoonBit writes its JavaScript entry point to `_build/js/release/build/mizchi/webnn-examples/app/app.js`, which [`public/index.html`](./public/index.html) loads.

## CI

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs on pushes, pull requests, manual dispatches, and every day at 03:17 JST to detect Chrome Canary updates. Permissions are limited to reading repository contents, and third-party Actions are pinned to commit SHAs.

- `MoonBit contracts`: CI configuration contracts, `moon fmt --check`, `moon check`, 117 unit tests, and a release build
- `WebNN Canary E2E`: 72 WebNN end-to-end tests on Ubuntu 24.04 with Playwright's `chromium-tip-of-tree` distribution (Chrome Canary for Testing headless shell)

On failure, Playwright HTML reports and traces are retained as artifacts for seven days. To reproduce the CI browser locally:

```bash
pnpm exec playwright install --only-shell chromium-tip-of-tree
WEBNN_BROWSER_CHANNEL=chromium-tip-of-tree just ci-e2e
```

The official MoonBit distribution uses rolling channels, so CI installs `latest` and records `moon version --all`. The daily run detects incompatibilities introduced by MoonBit updates as well.

Benchmark shapes, warmup counts, and iteration counts can be configured:

```bash
just bench 64,128,256 10 30
just bench-mnist 1,16,100 10 30
just bench-mnist-cache 1,16,100 10 30
just bench-mnist-pool 1,2,4,8 256 10 30
just bench-named-io 64,1024,16384 10 30
just bench-tiny-cnn 1,2,4,8,16,32,64 10 30 nchw,nhwc
just bench-transformer 32,64,128,256 16 10 30
just bench-attention 32,64,128,256 16 4 causal 10 30
just bench-encoder 32,64,128,256 2 16 4 causal-padding 10 30
just bench-encoder-stack 1,2,4 2 16 64 4 causal-padding 10 30
just bench-bert-encoder 1,2,4 2 16 64 4 padding 10 30
just bench-litert 10 30
just bench-tflite 10 30
just bench-tflite-runner-cache 10 30
just bench-mobilenet-v2 3 10
```

The measurement methodology and results from a local Apple M5 machine are recorded in [BENCHMARK.md](./BENCHMARK.md).

## Design

Model definitions use a tagless-final style inspired by MoonXi Net. `forward[T : TensorOps]` on `Linear`, `Mlp`, `TinyCnn`, `LayerNorm`, `FeedForward`, `SelfAttention`, `TransformerEncoderBlock/Stack`, and `BertEncoderBlock/Stack` is backend-independent, so the same model code runs with both `CpuTensor` and `WebNNTensor`. `TransformerEncoderBlock` defines pre-norm residual boundaries, while `BertEncoderBlock` uses the post-norm boundaries of Hugging Face BERT checkpoints. The existing `TransformerBlock` remains as a compatibility API for the older feed-forward benchmark.

```text
model/Linear
    |
    v
tensor/TensorOps
    |---------------------|
    v                     v
backend/cpu          backend/webnn
eager execution      builds MLOperands
                          |
                          v
                    compile / dispatch / read
```

Responsibilities are separated by layer:

- `mizchi/webnn`: curated public facade for the primary inference contracts
- `shape`: shape validation and result-shape inference
- `tensor`: backend-neutral operation contracts
- `model`: backend-independent model definitions
- `bert`: Hugging Face named-tensor validation, dense-weight transposition, and conversion to BERT parameters
- `backend/cpu`: eager CPU implementation for differential testing
- `webnn/raw`: thin FFI over the WebNN JavaScript API
- `webnn/compat`: compatibility between Chrome's current `deviceType` API and the newer `accelerated` contract
- `backend/webnn`: symbolic tensors, compilation, program cache, execution pools, and tensor lifecycle
- `examples/playground/src/app`: focused consumer-side browser API adapters called from Playwright
- `examples/playground/src/benchmark`: setup and steady-state measurements with CPU/WebNN correctness comparisons
- `litert`: source-neutral IR for named tensors, float32 constants, and operation sequences, plus a `TensorOps` lowerer
- `examples/playground/src/mnist`: trained MLP weights, IDX loading, and CPU/WebNN evaluation and benchmarks

The public Transformer API separates validated configuration, backend-independent parameters, and materialized models:

```text
SelfAttentionConfig / TransformerEncoderConfig / TransformerEncoderStackConfig
                         + Parameters (owned float32 arrays per layer)
                         + AttentionMask
                                     |
                    |----------------|----------------|
                    v                                 v
@cpu.materialize_transformer_encoder_stack   builder.materialize_transformer_encoder_stack
                    |                                 |
                    v                                 v
TransformerEncoderStack[CpuTensor]     TransformerEncoderStack[WebNNTensor]
```

`SelfAttentionConfig::new(width, heads)`, `TransformerEncoderConfig::new(width, heads, hidden_size, input_rank, epsilon)`, and `TransformerEncoderStackConfig::new(layers, ...)` validate dimensions and layer counts. `SelfAttentionParameters`, `FeedForwardParameters`, `TransformerEncoderParameters`, and `TransformerEncoderStackParameters` own copies of their input arrays, and their getters return copies. CPU and WebNN `materialize_transformer_encoder_stack()` implementations materialize the mask constant once and share it across all layers.

In addition to batch-shared `causal`, `padding`, and `causal_padding` masks, `AttentionMask` provides `batched_padding` and `batched_causal_padding` to generate `[batch,1,tokens,tokens]`. `additive(shape, values)` accepts any validated additive mask.

### BERT encoder loader

`BertEncoderConfig` and `BertEncoderParameters` define the post-norm BERT contract. Pass Hugging Face/PyTorch checkpoint tensor names, shapes, and float32 values as `@bert.NamedTensor`, then convert them with `load_hugging_face_encoder(config, tensors, "bert.encoder")`. Query/key/value, attention output, intermediate, and output dense weights are transposed from checkpoint `[out_features,in_features]` layout to the `Linear` `[in_features,out_features]` layout. Missing tensors, duplicates, and shape mismatches produce `TensorError` values containing the full tensor name. Tensors outside the encoder, such as embeddings and the pooler, are ignored.

`parse_safetensors(bytes)` validates the SafeTensors eight-byte little-endian header length, UTF-8 JSON header, data-section-relative offsets, and row-major little-endian F32 data, then returns `Array[NamedTensor]`. It also rejects gaps, overlapping offsets, unreferenced trailing bytes, and mismatches between shapes and byte lengths. `just fixture-bert` generates a deterministic two-layer, width-8, 32-tensor `.safetensors` file. Canary E2E covers `fetch → parse → mapping → CPU/WebNN`.

CPU and WebNN `materialize_bert_encoder_stack()` implementations materialize the padding mask once and share it across all layers. Dropout is omitted because this is an inference implementation.

`WebNNProgramCache` internally owns each `WebNNProgram`, which combines a compiled graph with prepared input and output tensors. Its public API only exposes `run_or_compile()` and `run_named_or_compile()`, so callers cannot accidentally `destroy()` a cached program. Cache keys are versioned canonical strings containing graph identity, device preference, ordered input shapes, and execution-pool size. The MNIST graph identity also includes the MLP topology/layout version and the SHA-256 of fixture weights, preventing reuse after weights or shapes change. If the same key is registered while another cache miss is compiling, the later program is destroyed and the existing entry is retained. Cached-resource cleanup is centralized in `clear()`. Clearing advances a generation, so an older in-flight compilation may return its result to the caller but is destroyed afterward and never reinserted.

`WebNNProgram` owns independent input/output tensor slots for its pool size, and a FIFO scheduler assigns runs to available slots. Pool size 1 serializes write/dispatch/read access to the same prepared tensors. Larger pools allow concurrent execution across slots. A failed run does not stop later runs. `destroy()` rejects new runs, waits for all accepted runs, and then releases every slot and its context.

The default pool size is 1 to favor sequential latency and memory use. In the concurrent MNIST measurements, size 2 captured most of the throughput improvement. Start with 2 for concurrent workloads, and benchmark before selecting 4 or more.

Graph I/O is represented by `WebNNInput` and `WebNNOutput`; runtime values use `WebNNNamedValues`. `compile_named()` and `run_named()` bind inputs by name rather than array order and return outputs in graph declaration order. Empty, duplicate, unknown, missing, and incorrectly sized values are rejected in MoonBit before WebNN dispatch. The older `compile_single()` and `run()` APIs remain as compatibility wrappers for one-input, one-output graphs.

The backend reads `opSupportLimits()` and fails before graph construction when a required operator is not reported. Shapes and input/constant element counts are also validated in MoonBit. `probe_capabilities(preference)` returns a `WebNNCapabilities` value containing the recognized context contract, preferred input layout, availability of `opSupportLimits()`, prepared tensor I/O support, and the backend's operator set. Canary E2E validates this value to detect specification or implementation changes.

`WebNNInput` and `WebNNOutput` keep names, shapes, and operands together, preventing operands from another builder from being passed during compilation. `WebNNGraphBuilder` owns the context. One-shot sessions use `defer graph.destroy()`; compiled programs release all slots and their context through `program.destroy()`.

`LiteRtModel` declares inputs, constants, and intermediates by name and shape, validates its operation sequence, and then lowers to generic `TensorOps`. It currently represents `add`, `sub`, `mul`, `div`, `reduce_mean`, `gather`, `slice`, `gelu`, `layer_normalization`, `concat`, `matmul`, `conv2d`, `max_pool2d`, `average_pool2d`, `sigmoid`, `tanh`, `clamp`, `relu`, `softmax`, `reshape`, and `transpose`. `gelu` uses the current WebNN exact definition, `0.5*x*(1+erf(x/sqrt(2)))`, as the shared CPU/WebNN contract. `layer_normalization(input, scale, bias, axes, epsilon)` requires explicit, non-empty, unique axes; the shapes of `scale` and `bias` must contain the corresponding dimensions in axis order, and epsilon must be positive. WebNN-specific `WebNNGraphBuilder.lower_litert(model)` materializes inputs and constants. `compile_litert_program(model)` and `compile_litert_program_pool(model, pool_size)` own lowering, named I/O binding, compilation, and prepared execution.

`TfliteModel::parse(bytes)` reads a FlatBuffer following the [TFLite schema](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/3e5424592ce7a5eeded530a58cc42f9fb981e40a/tensorflow/lite/schema/schema.fbs) and converts a single subgraph containing `ADD`, `SUB`, `MUL`, `DIV`, `MEAN`, `GATHER`, `SLICE`, `CONCATENATION`, `RELU`, `RELU_N1_TO_1`, `RELU6`, `LOGISTIC`, `TANH`, `SOFTMAX`, `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `RESHAPE`, or `TRANSPOSE` into float32 IR. `MEAN` accepts constant INT32 axes, `GATHER` accepts constant INT32 indices, and `SLICE` accepts constant INT32 begin/size values, where `-1` extends to the end. Axes and output shapes are validated. `CONCATENATION` lowers two or more inputs into binary `concat` steps and supports negative axes and fused activations. `UINT8` and `INT8` constants are statically dequantized with `scale * (q - zero_point)`. Per-axis metadata, the current `quantized_dimension` field position, older encoder layouts, and quantized `INT32` biases are supported. When a rank-1 bias preserves its producer's NHWC channel axis, it is normalized to axis 0 if the scale count matches the bias element count. Operator outputs are classified as intermediates from graph topology rather than buffer presence. `FULLY_CONNECTED` inputs above rank 2 are explicitly reshaped to `[batch, flattened]` before lowering. Custom operators are rejected by name before execution.

For arbitrary `.tflite` bytes, use the high-level `WebNNRuntime` or the lower-level `TfliteRunner` and `TfliteRunnerCache`. `WebNNRuntime::new(preference)` and `new_with_pool(preference, pool_size)` own a WebNN LRU cache limited to 32 models and 64 MiB of source FlatBuffers. Use `new_with_options(preference, pool_size, capacity, fallback)` to configure limits and fallback, `new_with_cpu_fallback(preference, pool_size, capacity)` to retry on CPU after WebNN failure, or `new_with_cache_limits(preference, pool_size, webnn_entries, webnn_bytes, cpu_entries, cpu_bytes, fallback)` to configure entry and byte budgets independently. Since browsers do not expose the exact memory used by compiled WebNN resources, byte budgets track the measurable source FlatBuffer size. Standalone caches provide `TfliteRunnerCache::new_with_limits(entries, bytes)` and `TfliteCpuRunnerCache::new_with_limits(entries, bytes)`. Models larger than the budget run transiently and are released immediately without evicting all existing entries.

`metrics()` returns cumulative WebNN cache hits, misses, evictions, entry count, and resident bytes, along with CPU fallback count and the corresponding parsed-model cache metrics. `clear()` releases WebNN programs and CPU parsed models without resetting counters. Its generation check prevents an in-flight compilation from reinserting itself after a clear. For steady-state execution of large models, `TfliteModelArtifact::from_bytes(bytes)` owns the bytes and computes the digest once; `run_prepared_tflite(artifact, inputs)` avoids recalculating SHA-256 on every call. CPU fallback uses `TfliteCpuRunner` with the same quantized I/O contract and reuses the parsed/lowered model by digest after the first call. Since WebNN is still attempted first each time, fallback is an availability policy rather than a normal-path performance substitute. Inputs are bound by TFLite tensor name. Float32 tensors accept `TfliteRunnerInput::float32`; quantized tensors accept `::quantized`. Outputs follow the original TFLite dtype and contain either float32 values or raw `Array[Int]` values.

## Real TFLite fixtures

`just fixture-tflite` downloads fixtures from pinned TensorFlow Lite Micro commits and verifies their SHA-256 digests.

- `micro_speech_quantized.tflite` (18,800 bytes) is a real quantized `RESHAPE → DEPTHWISE_CONV_2D → FULLY_CONNECTED → SOFTMAX` model. Canary E2E covers raw INT8 I/O, INT32 bias, flattening from rank 4, and runtime caching.
- `person_detect.tflite` (300,568 bytes) is a quantized 96×96 grayscale person-detection CNN. It exercises normalization of rank-1 per-channel INT32 bias metadata that retains producer axis 3, zero-input raw INT8 I/O, and the compiled runtime cache.
- `mobilenet_v2_1.0_224_inat_bird_quant.tflite` (3,531,296 bytes) is a quantized MobileNet V2 classification model from Coral Edge TPU. Canary E2E covers parse/lower/compile without a delegate, raw UINT8 I/O, and cache hits.
- `audio_preprocessor_int8.tflite` (8,772 bytes) is a negative fixture containing the `SignalWindow` custom operator. It currently locks in the rejection message `unsupported TFLite custom operator: SignalWindow`.

## Browser API

The built page exposes a test API as `globalThis.webnnPlayground`:

- `compatibilityMode()`
- `supportedOperators()`
- `probeWebnnCapabilities()`
- `runCpuLinear()`
- `runAdd()`
- `runConcat()`
- `runLinear()`
- `runLiteRtLinear()`
- `runTfliteAdd()`
- `runTfliteSub()`
- `runTfliteAddRelu6()`
- `runTfliteFullyConnected()`
- `runTfliteFullyConnectedRelu6()`
- `runTfliteConv2d()`
- `runTfliteReshape()`
- `runTfliteTranspose()`
- `runTfliteDepthwiseConv2d()`
- `runTfliteAveragePool2d()`
- `runTfliteMaxPool2d()`
- `runTfliteLogistic()`
- `runTfliteTanh()`
- `runTfliteQuantizedAdd()`
- `runTfliteQuantizedAddRaw()`
- `runTflitePerAxisInt8Raw()`
- `runTfliteReluN1To1()`
- `runTfliteRelu6()`
- `benchmarkLiteRtLinear(warmup, iterations)`
- `benchmarkTfliteFullyConnected(warmup, iterations)`
- `benchmarkTfliteSub(warmup, iterations)`
- `benchmarkTfliteDepthwiseConv2d(warmup, iterations)`
- `benchmarkTfliteMaxPool2d(warmup, iterations)`
- `benchmarkTflitePerAxisInt8Raw(warmup, iterations)`
- `runTfliteRunnerCachePerAxisInt8Raw()`
- `runTfliteRuntimeCachePerAxisInt8Raw()`
- `runTfliteCpuPerAxisInt8Raw()`
- `runTfliteRuntimeCpuFallbackPerAxisInt8Raw()`
- `probeTfliteRuntimePolicy()`
- `runMicroSpeechFixture()`
- `runPersonDetectionFixture()`
- `probeMobileNetV2Fixture()`
- `runMobileNetV2Fixture()`
- `benchmarkMobileNetV2Fixture(warmup, iterations)`
- `probeAudioPreprocessorFixture()`
- `probeTfliteRunnerCacheConcurrency()`
- `runTfliteRunnerNamedAdd()`
- `runTfliteRunnerTwoInputTwoOutput()`
- `benchmarkTfliteRunnerCachePerAxisInt8Raw(warmup, iterations)`
- `runTransformPipeline()`
- `benchmarkMatmul(size, warmup, iterations)`
- `benchmarkTransformerBlock(tokens, width, warmup, iterations)`
- `benchmarkSelfAttention(tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkTransformerEncoder(batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkTransformerEncoderStack(layers, batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkBertEncoder(layers, batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `runBertSafetensorsFixture()`
- `evaluateMnist(count)`
- `benchmarkMnist(batchSize, warmup, iterations)`
- `clearMnistCache()`
- `runMnistCached(batchSize)`
- `benchmarkMnistCache(batchSize, warmup, iterations)`
- `probeConcurrentProgram()` (E2E probe for concurrent runs and deferred destruction)
- `probeExecutionPool()` (E2E probe for multi-slot assignment)
- `benchmarkMnistPool(poolSize, requestCount, warmup, iterations)`
- `runNamedMultiIo()`
- `benchmarkNamedIo(size, warmup, iterations)`
- `runConvSoftmax()`
- `benchmarkTinyCnn(batchSize, warmup, iterations)`
- `benchmarkTinyCnnLayout(inputLayout, batchSize, warmup, iterations)`
- `runPreferredLayoutConv()`

`runLinear()` and `runCpuLinear()` use the same generic `Linear.forward` implementation, and Playwright compares their results.

## Limitations

- Not every WebNN operator, data type, or dynamic shape is supported.
- The `.tflite` parser is limited to a single subgraph and `ADD`/`SUB`/`MUL`/`DIV`/`MEAN`/`GATHER`/`SLICE`/`CONCATENATION`/`RELU`/`RELU_N1_TO_1`/`RELU6`/`LOGISTIC`/`TANH`/`SOFTMAX`/`FULLY_CONNECTED`/`CONV_2D`/`DEPTHWISE_CONV_2D`/`AVERAGE_POOL_2D`/`MAX_POOL_2D`/`RESHAPE`/`TRANSPOSE`. `GELU` and `layer_normalization` are available in the generic IR/API but are not yet emitted by the TFLite parser. `CONCATENATION` supports negative axes, two or more inputs, and fused activations. `UINT8` and `INT8` require positive scales and zero points; constants are statically dequantized to float32 using either scalar per-tensor or shape-compatible per-axis quantization. Quantized `INT32` is supported only for bias constants; int32 computation and runtime inputs are not supported. Fused activations are limited to `NONE`, `RELU`, `RELU_N1_TO_1`, `RELU6`, and `TANH`; `SIGN_BIT` is not supported. `INT16` and `FLOAT16` are not supported. Multiple subgraphs, control flow, and custom operators such as `SignalWindow` are not supported.
- `conv2d` supports NCHW inputs with OIHW filters and NHWC inputs with HWIO filters. Padding, stride, dilation, and groups are supported. Bias is expressed as an `add` with `[1,C,1,1]` for NCHW or `[1,1,1,C]` for NHWC. `WebNNGraphBuilder.preferred_input_layout()` reads `opSupportLimits().preferredInputLayout` to guide layout selection.
- The `accelerated` contract distinguishes only CPU from accelerated execution, so `Npu` maps to `accelerated: true`.
- Even when the current Canary implementation accepts `deviceType: "npu"`, the WebNN API cannot confirm that Apple Neural Engine was selected.
- A native backend is intentionally outside this implementation. It may be added later as a Core ML or ONNX Runtime backend.
- The BERT loader covers encoder layers only. Embeddings, poolers, task heads, ONNX, and `gelu_new` are unsupported. The SafeTensors parser accepts only a single file containing F32 tensors with positive fixed shapes; F16/BF16, sharded indexes, and zero-sized tensors are unsupported.
- Execution pools share the same browser thread and WebNN context. Increasing pool size cannot exceed the concurrency available in the hardware/backend and increases per-slot tensor memory.
- In Canary 152, passing the same `MLOperand` directly under two output names caused the build to fail with `Context is lost`. Creating a distinct operand with `reshape` avoids the issue.

References:

- [WebNN specification](https://www.w3.org/TR/webnn/)
- [SafeTensors format](https://github.com/huggingface/safetensors#format)
- [MoonXi Net](https://www.moonbitlang.com/pearls/moonxi-net)
- [Playwright browser channels](https://playwright.dev/docs/browsers)
