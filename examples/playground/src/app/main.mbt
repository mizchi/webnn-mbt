///|
/// Browser playground entry point built as a consumer of `mizchi/webnn`.
extern "js" fn expose_api(
  compatibility_mode : () -> String,
  supported_operators : () -> @js.Promise[Array[String]],
  probe_webnn_capabilities : () -> @js.Promise[@core.Any],
  run_cpu_linear : () -> @js.Promise[Array[Float]],
  run_add : () -> @js.Promise[Array[Float]],
  run_concat : () -> @js.Promise[Array[Float]],
  run_linear : () -> @js.Promise[Array[Float]],
  run_litert_linear : () -> @js.Promise[Array[Float]],
  run_tflite_add : () -> @js.Promise[Array[Float]],
  run_tflite_add_relu6 : () -> @js.Promise[Array[Float]],
  run_tflite_fully_connected : () -> @js.Promise[Array[Float]],
  run_tflite_fully_connected_relu6 : () -> @js.Promise[Array[Float]],
  run_tflite_conv2d : () -> @js.Promise[Array[Float]],
  run_tflite_reshape : () -> @js.Promise[Array[Float]],
  run_tflite_transpose : () -> @js.Promise[Array[Float]],
  run_tflite_depthwise_conv2d : () -> @js.Promise[Array[Float]],
  run_tflite_average_pool2d : () -> @js.Promise[Array[Float]],
  run_tflite_max_pool2d : () -> @js.Promise[Array[Float]],
  run_tflite_logistic : () -> @js.Promise[Array[Float]],
  run_tflite_tanh : () -> @js.Promise[Array[Float]],
  run_tflite_quantized_add : () -> @js.Promise[Array[Float]],
  run_tflite_quantized_add_raw : () -> @js.Promise[Array[Int]],
  run_tflite_per_axis_int8_raw : () -> @js.Promise[Array[Int]],
  run_tflite_runner_cache_per_axis_int8_raw : () -> @js.Promise[@core.Any],
  run_tflite_runtime_cache_per_axis_int8_raw : () -> @js.Promise[@core.Any],
  run_tflite_cpu_per_axis_int8_raw : () -> @js.Promise[Array[Int]],
  run_tflite_runtime_cpu_fallback_per_axis_int8_raw : () -> @js.Promise[
    @core.Any,
  ],
  probe_tflite_runtime_policy : () -> @js.Promise[@core.Any],
  probe_tflite_runtime_cpu_fallback_cache : () -> @js.Promise[@core.Any],
  probe_tflite_runtime_byte_budget : () -> @js.Promise[@core.Any],
  run_micro_speech_fixture : () -> @js.Promise[@core.Any],
  run_person_detection_fixture : () -> @js.Promise[@core.Any],
  probe_mobilenet_v2_fixture : () -> @js.Promise[@core.Any],
  run_mobilenet_v2_fixture : () -> @js.Promise[@core.Any],
  benchmark_mobilenet_v2_fixture : (Int, Int) -> @js.Promise[@core.Any],
  probe_audio_preprocessor_fixture : () -> @js.Promise[@core.Any],
  probe_tflite_runner_cache_concurrency : () -> @js.Promise[@core.Any],
  probe_tflite_runner_cache_clear_during_compile : () -> @js.Promise[@core.Any],
  run_tflite_runner_named_add : () -> @js.Promise[Array[Float]],
  run_tflite_runner_two_input_two_output : () -> @js.Promise[@core.Any],
  run_tflite_sub : () -> @js.Promise[Array[Float]],
  run_tflite_relu_n1_to_1 : () -> @js.Promise[Array[Float]],
  run_tflite_relu6 : () -> @js.Promise[Array[Float]],
  benchmark_litert_linear : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_fully_connected : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_sub : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_depthwise_conv2d : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_max_pool2d : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_per_axis_int8_raw : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_runner_cache_per_axis_int8_raw : (Int, Int) -> @js.Promise[
    @core.Any,
  ],
  run_transform_pipeline : () -> @js.Promise[Array[Float]],
  benchmark_matmul : (Int, Int, Int) -> @js.Promise[@core.Any],
  benchmark_transformer_block : (Int, Int, Int, Int) -> @js.Promise[@core.Any],
  benchmark_self_attention : (Int, Int, Int, String, Int, Int) -> @js.Promise[
    @core.Any,
  ],
  benchmark_transformer_encoder : (Int, Int, Int, Int, String, Int, Int) -> @js.Promise[
    @core.Any,
  ],
  benchmark_transformer_encoder_stack : (
    Int,
    Int,
    Int,
    Int,
    Int,
    String,
    Int,
    Int,
  ) -> @js.Promise[@core.Any],
  benchmark_bert_encoder : (Int, Int, Int, Int, Int, String, Int, Int) -> @js.Promise[
    @core.Any,
  ],
  run_bert_safetensors_fixture : () -> @js.Promise[@core.Any],
  evaluate_mnist : (Int) -> @js.Promise[@core.Any],
  benchmark_mnist : (Int, Int, Int) -> @js.Promise[@core.Any],
  clear_mnist_cache : () -> Int,
  run_mnist_cached : (Int) -> @js.Promise[@core.Any],
  benchmark_mnist_cache : (Int, Int, Int) -> @js.Promise[@core.Any],
  probe_program_cache_clear_during_compile : () -> @js.Promise[@core.Any],
  probe_concurrent_program : () -> @js.Promise[@core.Any],
  probe_execution_pool : () -> @js.Promise[@core.Any],
  benchmark_mnist_pool : (Int, Int, Int, Int) -> @js.Promise[@core.Any],
  run_named_multi_io : () -> @js.Promise[@core.Any],
  benchmark_named_io : (Int, Int, Int) -> @js.Promise[@core.Any],
  run_conv_softmax : () -> @js.Promise[Array[Float]],
  benchmark_tiny_cnn : (Int, Int, Int) -> @js.Promise[@core.Any],
  benchmark_tiny_cnn_layout : (String, Int, Int, Int) -> @js.Promise[@core.Any],
  run_grouped_conv : () -> @js.Promise[Array[Float]],
  run_preferred_layout_conv : () -> @js.Promise[@core.Any],
) -> Unit =
  #| (compatibilityMode, supportedOperators, probeWebnnCapabilities, runCpuLinear, runAdd, runConcat, runLinear, runLiteRtLinear, runTfliteAdd, runTfliteAddRelu6, runTfliteFullyConnected, runTfliteFullyConnectedRelu6, runTfliteConv2d, runTfliteReshape, runTfliteTranspose, runTfliteDepthwiseConv2d, runTfliteAveragePool2d, runTfliteMaxPool2d, runTfliteLogistic, runTfliteTanh, runTfliteQuantizedAdd, runTfliteQuantizedAddRaw, runTflitePerAxisInt8Raw, runTfliteRunnerCachePerAxisInt8Raw, runTfliteRuntimeCachePerAxisInt8Raw, runTfliteCpuPerAxisInt8Raw, runTfliteRuntimeCpuFallbackPerAxisInt8Raw, probeTfliteRuntimePolicy, probeTfliteRuntimeCpuFallbackCache, probeTfliteRuntimeByteBudget, runMicroSpeechFixture, runPersonDetectionFixture, probeMobileNetV2Fixture, runMobileNetV2Fixture, benchmarkMobileNetV2Fixture, probeAudioPreprocessorFixture, probeTfliteRunnerCacheConcurrency, probeTfliteRunnerCacheClearDuringCompile, runTfliteRunnerNamedAdd, runTfliteRunnerTwoInputTwoOutput, runTfliteSub, runTfliteReluN1To1, runTfliteRelu6, benchmarkLiteRtLinear, benchmarkTfliteFullyConnected, benchmarkTfliteSub, benchmarkTfliteDepthwiseConv2d, benchmarkTfliteMaxPool2d, benchmarkTflitePerAxisInt8Raw, benchmarkTfliteRunnerCachePerAxisInt8Raw, runTransformPipeline, benchmarkMatmul, benchmarkTransformerBlock, benchmarkSelfAttention, benchmarkTransformerEncoder, benchmarkTransformerEncoderStack, benchmarkBertEncoder, runBertSafetensorsFixture, evaluateMnist, benchmarkMnist, clearMnistCache, runMnistCached, benchmarkMnistCache, probeProgramCacheClearDuringCompile, probeConcurrentProgram, probeExecutionPool, benchmarkMnistPool, runNamedMultiIo, benchmarkNamedIo, runConvSoftmax, benchmarkTinyCnn, benchmarkTinyCnnLayout, runGroupedConv, runPreferredLayoutConv) => {
  #|   globalThis.webnnPlayground = {
  #|     compatibilityMode,
  #|     supportedOperators,
  #|     probeWebnnCapabilities,
  #|     runCpuLinear,
  #|     runAdd,
  #|     runConcat,
  #|     runLinear,
  #|     runLiteRtLinear,
  #|     runTfliteAdd,
  #|     runTfliteAddRelu6,
  #|     runTfliteFullyConnected,
  #|     runTfliteFullyConnectedRelu6,
  #|     runTfliteConv2d,
  #|     runTfliteReshape,
  #|     runTfliteTranspose,
  #|     runTfliteDepthwiseConv2d,
  #|     runTfliteAveragePool2d,
  #|     runTfliteMaxPool2d,
  #|     runTfliteLogistic,
  #|     runTfliteTanh,
  #|     runTfliteQuantizedAdd,
  #|     runTfliteQuantizedAddRaw,
  #|     runTflitePerAxisInt8Raw,
  #|     runTfliteRunnerCachePerAxisInt8Raw,
  #|     runTfliteRuntimeCachePerAxisInt8Raw,
  #|     runTfliteCpuPerAxisInt8Raw,
  #|     runTfliteRuntimeCpuFallbackPerAxisInt8Raw,
  #|     probeTfliteRuntimePolicy,
  #|     probeTfliteRuntimeCpuFallbackCache,
  #|     probeTfliteRuntimeByteBudget,
  #|     runMicroSpeechFixture,
  #|     runPersonDetectionFixture,
  #|     probeMobileNetV2Fixture,
  #|     runMobileNetV2Fixture,
  #|     benchmarkMobileNetV2Fixture,
  #|     probeAudioPreprocessorFixture,
  #|     probeTfliteRunnerCacheConcurrency,
  #|     probeTfliteRunnerCacheClearDuringCompile,
  #|     runTfliteRunnerNamedAdd,
  #|     runTfliteRunnerTwoInputTwoOutput,
  #|     runTfliteSub,
  #|     runTfliteReluN1To1,
  #|     runTfliteRelu6,
  #|     benchmarkLiteRtLinear,
  #|     benchmarkTfliteFullyConnected,
  #|     benchmarkTfliteSub,
  #|     benchmarkTfliteDepthwiseConv2d,
  #|     benchmarkTfliteMaxPool2d,
  #|     benchmarkTflitePerAxisInt8Raw,
  #|     benchmarkTfliteRunnerCachePerAxisInt8Raw,
  #|     runTransformPipeline,
  #|     benchmarkMatmul,
  #|     benchmarkTransformerBlock,
  #|     benchmarkSelfAttention,
  #|     benchmarkTransformerEncoder,
  #|     benchmarkTransformerEncoderStack,
  #|     benchmarkBertEncoder,
  #|     runBertSafetensorsFixture,
  #|     evaluateMnist,
  #|     benchmarkMnist,
  #|     clearMnistCache,
  #|     runMnistCached,
  #|     benchmarkMnistCache,
  #|     probeProgramCacheClearDuringCompile,
  #|     probeConcurrentProgram,
  #|     probeExecutionPool,
  #|     benchmarkMnistPool,
  #|     runNamedMultiIo,
  #|     benchmarkNamedIo,
  #|     runConvSoftmax,
  #|     benchmarkTinyCnn,
  #|     benchmarkTinyCnnLayout,
  #|     runGroupedConv,
  #|     runPreferredLayoutConv,
  #|   };
  #| }

///|
fn run_grouped_conv() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_binding = builder.input("input", @shape.Shape::new([1, 2, 2, 2]))
    let filter = builder.constant(@shape.Shape::new([2, 1, 1, 1]), [2.0, 3.0])
    let options = @shape.Conv2dOptions::new([0, 0, 0, 0], [1, 1], [1, 1], 2)
    let output = input_binding.tensor().conv2d(filter, options)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  })
}

///|
extern "js" fn make_preferred_layout_conv_report(
  preferred_input_layout : String,
  input_shape : Array[Int],
  filter_shape : Array[Int],
  output : Array[Float],
) -> @core.Any =
  #| (preferredInputLayout, inputShape, filterShape, output) => ({
  #|   preferredInputLayout,
  #|   inputShape,
  #|   filterShape,
  #|   output,
  #| })

///|
fn run_preferred_layout_conv() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_layout = builder.preferred_input_layout()
    let (input_shape, filter_shape, filter_layout) = match input_layout {
      @shape.InputLayout::Nchw =>
        (
          @shape.Shape::new([1, 1, 3, 3]),
          @shape.Shape::new([1, 1, 2, 2]),
          @shape.Conv2dFilterLayout::Oihw,
        )
      @shape.InputLayout::Nhwc =>
        (
          @shape.Shape::new([1, 3, 3, 1]),
          @shape.Shape::new([2, 2, 1, 1]),
          @shape.Conv2dFilterLayout::Hwio,
        )
    }
    let input_binding = builder.input("input", input_shape)
    let filter = builder.constant(filter_shape, [1.0, 0.0, 0.0, -1.0])
    let options = @shape.Conv2dOptions::new_with_layout(
      [0, 0, 0, 0],
      [1, 1],
      [1, 1],
      1,
      input_layout,
      filter_layout,
    )
    let output = input_binding.tensor().conv2d(filter, options)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    make_preferred_layout_conv_report(
      input_layout.to_webnn_string(),
      input_shape.dimensions(),
      filter_shape.dimensions(),
      program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )
  })
}

///|
fn benchmark_tiny_cnn(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.tiny_cnn(batch_size, warmup, iterations) })
}

///|
fn benchmark_tiny_cnn_layout(
  input_layout : String,
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let layout = match input_layout {
      "nchw" => @shape.InputLayout::Nchw
      "nhwc" => @shape.InputLayout::Nhwc
      _ =>
        raise @tensor.TensorError::new(
          "Tiny CNN input layout must be nchw or nhwc",
        )
    }
    @bench.tiny_cnn_with_layout(batch_size, warmup, iterations, layout)
  })
}

///|
fn run_conv_softmax() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_binding = builder.input("input", @shape.Shape::new([1, 1, 3, 3]))
    let filter = builder.constant(@shape.Shape::new([1, 1, 2, 2]), [
      1.0, 0.0, 0.0, 0.0,
    ])
    let output = input_binding
      .tensor()
      .conv2d(filter, @shape.Conv2dOptions::default())
      .softmax(3)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  })
}

///|
fn benchmark_named_io(
  size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.named_io(size, warmup, iterations) })
}

///|
extern "js" fn make_named_multi_io_report(
  names : Array[String],
  values : Array[Array[Float]],
  duplicate_input_rejected : Bool,
  unknown_input_rejected : Bool,
  input_count_rejected : Bool,
  input_length_rejected : Bool,
  duplicate_output_rejected : Bool,
) -> @core.Any =
  #| (names, values, duplicateInputRejected, unknownInputRejected, inputCountRejected, inputLengthRejected, duplicateOutputRejected) => ({
  #|   names,
  #|   outputs: Object.fromEntries(
  #|     names.map((name, index) => [name, values[index]]),
  #|   ),
  #|   duplicateInputRejected,
  #|   unknownInputRejected,
  #|   inputCountRejected,
  #|   inputLengthRejected,
  #|   duplicateOutputRejected,
  #| })

///|
fn run_named_multi_io() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let shape = @shape.Shape::new([2])
    let x = builder.input("x", shape)
    let y = builder.input("y", shape)
    let product = x.tensor().mul(y.tensor())
    let sum = x.tensor().add(y.tensor())
    let duplicate_output_rejected = try
      builder.compile_named([x, y], [
        builder.output("duplicate", product),
        builder.output("duplicate", sum),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let program = builder.compile_program_pool_named(
      [x, y],
      [builder.output("product", product), builder.output("sum", sum)],
      2,
    )
    defer program.destroy()
    let duplicate_input_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
        @webnn.WebNNNamedValues::new("x", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let unknown_input_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
        @webnn.WebNNNamedValues::new("z", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let input_count_rejected = try
      program.run_named([@webnn.WebNNNamedValues::new("x", [1.0, 2.0])])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let input_length_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0]),
        @webnn.WebNNNamedValues::new("y", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let results = program.run_named([
      @webnn.WebNNNamedValues::new("y", [3.0, 4.0]),
      @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
    ])
    make_named_multi_io_report(
      results.map(fn(result) { result.name() }),
      results.map(fn(result) { result.values() }),
      duplicate_input_rejected,
      unknown_input_rejected,
      input_count_rejected,
      input_length_rejected,
      duplicate_output_rejected,
    )
  })
}

///|
fn benchmark_mnist_pool(
  pool_size : Int,
  request_count : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @mnist.benchmark_pool(pool_size, request_count, warmup, iterations)
  })
}

///|
extern "js" fn run_program_pair(
  run : (Array[Float]) -> @js.Promise[Array[Float]],
  first : Array[Float],
  second : Array[Float],
) -> @js.Promise[Array[Array[Float]]] =
  #| (run, first, second) => Promise.all([run(first), run(second)])

///|
extern "js" fn run_int_program_pair(
  run : (Array[Int]) -> @js.Promise[Array[Int]],
  first : Array[Int],
  second : Array[Int],
) -> @js.Promise[Array[Array[Int]]] =
  #| (run, first, second) => Promise.all([run(first), run(second)])

///|
extern "js" fn make_program_concurrency_report(
  results : Array[Array[Float]],
  rejected_run_recovered : Bool,
  result_after_rejected_run : Array[Float],
  run_after_destroy_rejected : Bool,
) -> @core.Any =
  #| (results, rejectedRunRecovered, resultAfterRejectedRun, runAfterDestroyRejected) => ({
  #|   results,
  #|   rejectedRunRecovered,
  #|   resultAfterRejectedRun,
  #|   runAfterDestroyRejected,
  #| })

///|
extern "js" fn make_tflite_runner_cache_concurrency_report(
  results : Array[Array[Int]],
  cache_size : Int,
) -> @core.Any =
  #| (results, cacheSize) => ({ results, cacheSize })

///|
extern "js" fn make_execution_pool_report(
  results : Array[Array[Float]],
  pool_size : Int,
  maximum_concurrency : Int,
  run_after_destroy_rejected : Bool,
) -> @core.Any =
  #| (results, poolSize, maximumConcurrency, runAfterDestroyRejected) => ({
  #|   results,
  #|   poolSize,
  #|   maximumConcurrency,
  #|   runAfterDestroyRejected,
  #| })

///|
async fn compile_probe_program(pool_size : Int) -> @webnn.WebNNProgram {
  let input_shape = @shape.Shape::new([1, 2])
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let input_binding = builder.input("input", input_shape)
  let weight = builder.constant(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
  let bias = builder.constant(@shape.Shape::new([2]), [1.0, 1.0])
  let output = @model.Linear::{ weight, bias }.forward(input_binding.tensor())
  builder.compile_program_pool_single(
    input_binding, "output", output, pool_size,
  )
}

///|
#external
type CompileGate

///|
extern "js" fn new_compile_gate() -> CompileGate =
  #| () => {
  #|   let started;
  #|   let release;
  #|   return {
  #|     started: new Promise((resolve) => { started = resolve; }),
  #|     release: new Promise((resolve) => { release = resolve; }),
  #|     signalStarted: () => started(),
  #|     open: () => release(),
  #|   };
  #| }

///|
extern "js" fn compile_gate_signal_started(gate : CompileGate) -> Unit =
  #| (gate) => gate.signalStarted()

///|
extern "js" fn compile_gate_started(gate : CompileGate) -> @js.Promise[Unit] =
  #| (gate) => gate.started

///|
extern "js" fn compile_gate_wait(gate : CompileGate) -> @js.Promise[Unit] =
  #| (gate) => gate.release

///|
extern "js" fn compile_gate_open(gate : CompileGate) -> Unit =
  #| (gate) => gate.open()

///|
extern "js" fn make_program_cache_clear_race_report(
  output : Array[Float],
  cache_hit : Bool,
  cache_size_after_clear : Int,
) -> @core.Any =
  #| (output, cacheHit, cacheSizeAfterClear) => ({
  #|   output,
  #|   cacheHit,
  #|   cacheSizeAfterClear,
  #| })

///|
fn probe_program_cache_clear_during_compile() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let cache = @webnn.WebNNProgramCache::new()
    let key = @webnn.WebNNProgramCacheKey::new(
      "cache-clear-race-v1",
      @compat.DevicePreference::Npu,
      [@shape.Shape::new([1, 2])],
    )
    let gate = new_compile_gate()
    let pending = @js.from_async(async fn() {
      cache.run_or_compile(key, [1.0, 2.0], fn() {
        @js.from_async(async fn() {
          compile_gate_signal_started(gate)
          compile_gate_wait(gate).wait()
          compile_probe_program(1)
        })
      })
    })
    compile_gate_started(gate).wait()
    cache.clear() |> ignore
    compile_gate_open(gate)
    let (output, cache_hit) = pending.wait()
    make_program_cache_clear_race_report(output, cache_hit, cache.length())
  })
}

///|
extern "js" fn make_tflite_runner_cache_clear_race_report(
  output : Array[Float],
  cache_size_after_clear : Int,
  resident_bytes_after_clear : Int,
) -> @core.Any =
  #| (output, cacheSizeAfterClear, residentBytesAfterClear) => ({
  #|   output,
  #|   cacheSizeAfterClear,
  #|   residentBytesAfterClear,
  #| })

///|
/// Verifies that clear() invalidates an in-flight TFLite runner compilation,
/// so the completed runner is used once but never retained afterwards.
fn probe_tflite_runner_cache_clear_during_compile() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = tflite_add_fixture()
    let artifact = @webnn.TfliteModelArtifact::from_bytes(bytes)
    let cache = @webnn.TfliteRunnerCache::new()
    defer ignore(cache.clear())
    let gate = new_compile_gate()
    let pending = @js.from_async(async fn() {
      cache.run_artifact_pool_with_compiler(
        artifact,
        @compat.DevicePreference::Npu,
        1,
        [@webnn.TfliteRunnerInput::float32("input", [1.0, 2.0])],
        fn() {
          @js.from_async(async fn() {
            compile_gate_signal_started(gate)
            compile_gate_wait(gate).wait()
            @webnn.TfliteRunner::compile_pool(
              bytes,
              @compat.DevicePreference::Npu,
              1,
            )
          })
        },
      )
    })
    compile_gate_started(gate).wait()
    cache.clear() |> ignore
    compile_gate_open(gate)
    let outputs = pending.wait()
    if outputs.length() != 1 {
      raise @tensor.TensorError::new(
        "expected exactly one TFLite runner output",
      )
    }
    let output = match outputs[0].values() {
      @webnn.TfliteRunnerValues::Float32(values) => values
      @webnn.TfliteRunnerValues::Quantized(_) =>
        raise @tensor.TensorError::new(
          "expected a float32 TFLite runner output",
        )
    }
    make_tflite_runner_cache_clear_race_report(
      output,
      cache.length(),
      cache.resident_bytes(),
    )
  })
}

///|
fn probe_execution_pool() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let program = compile_probe_program(2)
    let pending = run_program_pair(
      fn(values) { @js.from_async(async fn() { program.run(values) }) },
      [1.0, 2.0],
      [3.0, 4.0],
    )
    program.destroy()
    let results = pending.wait()
    let run_after_destroy_rejected = try program.run([1.0, 2.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    make_execution_pool_report(
      results,
      program.pool_size(),
      program.maximum_concurrency(),
      run_after_destroy_rejected,
    )
  })
}

///|
fn probe_concurrent_program() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let program = compile_probe_program(1)
    let rejected_run_recovered = try program.run([1.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    let result_after_rejected_run = program.run([1.0, 2.0])
    let pending = run_program_pair(
      fn(values) { @js.from_async(async fn() { program.run(values) }) },
      [1.0, 2.0],
      [3.0, 4.0],
    )
    program.destroy()
    let results = pending.wait()
    let run_after_destroy_rejected = try program.run([1.0, 2.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    make_program_concurrency_report(
      results, rejected_run_recovered, result_after_rejected_run, run_after_destroy_rejected,
    )
  })
}

///|
fn benchmark_mnist_cache(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @mnist.benchmark_cache(batch_size, warmup, iterations)
  })
}

///|
fn run_mnist_cached(batch_size : Int) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.run_cached(batch_size) })
}

///|
fn evaluate_mnist(count : Int) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.evaluate(count) })
}

///|
fn benchmark_mnist(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.benchmark(batch_size, warmup, iterations) })
}

///|
fn benchmark_matmul(
  size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.matmul(size, warmup, iterations) })
}

///|
fn benchmark_transformer_block(
  tokens : Int,
  width : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_block(tokens, width, warmup, iterations)
  })
}

///|
fn benchmark_self_attention(
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.self_attention(tokens, width, heads, mask_mode, warmup, iterations)
  })
}

///|
fn benchmark_transformer_encoder(
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_encoder(
      batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
fn benchmark_transformer_encoder_stack(
  layers : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_encoder_stack(
      layers, batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
fn benchmark_bert_encoder(
  layers : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.bert_encoder(
      layers, batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
extern "js" fn make_bert_safetensors_fixture_report(
  file_bytes : Int,
  tensor_count : Int,
  layers : Int,
  input_shape : Array[Int],
  output_shape : Array[Int],
  cpu_output : Array[Float],
  webnn_output : Array[Float],
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (fileBytes, tensorCount, layers, inputShape, outputShape, cpuOutput, webnnOutput, maxAbsError) => ({
  #|   fileBytes,
  #|   tensorCount,
  #|   layers,
  #|   inputShape,
  #|   outputShape,
  #|   cpuOutput,
  #|   webnnOutput,
  #|   maxAbsError,
  #| })

///|
fn bert_fixture_max_abs_error(
  expected : Array[Float],
  actual : Array[Float],
) -> Float {
  let mut maximum : Float = 0.0
  for index, value in expected {
    let difference = (value - actual[index]).abs()
    if difference > maximum {
      maximum = difference
    }
  }
  maximum
}

///|
fn run_bert_safetensors_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/bert/tiny-bert-encoder.safetensors",
    ).wait()
    let tensors = @bert.parse_safetensors(bytes)
    let config = @model.BertEncoderConfig::new(2, 8, 2, 16, 3, 1.0e-12)
    let parameters = @bert.load_hugging_face_encoder(
      config, tensors, "bert.encoder",
    )
    let input_shape = @shape.Shape::new([2, 4, 8])
    let inputs = Array::makei(input_shape.element_count(), fn(index) {
      Float::from_int((index * 7 + index / 5) % 37 - 18) / Float::from_int(16)
    })
    let mask = @model.AttentionMask::batched_padding(
      [[true, true, true, true], [true, true, true, false]],
      -10000.0,
    )
    let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
    let cpu_output = @cpu.materialize_bert_encoder_stack(
        config,
        parameters,
        Some(mask),
      )
      .forward(cpu_input)
      .values()
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer builder.destroy()
    let input_binding = builder.input("input", input_shape)
    let output = builder
      .materialize_bert_encoder_stack(config, parameters, Some(mask))
      .forward(input_binding.tensor())
    let session = builder.compile_single(input_binding, "output", output)
    let webnn_output = session.run(inputs)
    make_bert_safetensors_fixture_report(
      bytes.length(),
      tensors.length(),
      config.layers(),
      input_shape.dimensions(),
      output.shape().dimensions(),
      cpu_output,
      webnn_output,
      bert_fixture_max_abs_error(cpu_output, webnn_output),
    )
  })
}

///|
fn supported_operators() -> @js.Promise[Array[String]] {
  @js.from_async(async fn() {
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    graph.supported_operators()
  })
}

///|
extern "js" fn make_webnn_capabilities_report(
  compatibility_mode : String,
  preferred_input_layout : String,
  op_support_limits_available : Bool,
  tensor_io_available : Bool,
  supported_operators : Array[String],
) -> @core.Any =
  #| (compatibilityMode, preferredInputLayout, opSupportLimitsAvailable, tensorIoAvailable, supportedOperators) => ({
  #|   compatibilityMode,
  #|   preferredInputLayout,
  #|   opSupportLimitsAvailable,
  #|   tensorIoAvailable,
  #|   supportedOperators,
  #| })

///|
fn probe_webnn_capabilities() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let capabilities = @webnn.probe_capabilities(@compat.DevicePreference::Npu)
    make_webnn_capabilities_report(
      capabilities.contract().to_string(),
      capabilities.preferred_input_layout().to_webnn_string(),
      capabilities.op_support_limits_available(),
      capabilities.tensor_io_available(),
      capabilities.supported_operators(),
    )
  })
}

///|
fn run_cpu_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(fn() raise {
    let input = @cpu.CpuTensor::new(@shape.Shape::new([1, 2]), [1.0, 2.0])
    let weight = @cpu.CpuTensor::new(@shape.Shape::new([2, 2]), [
      1.0, 2.0, 3.0, 4.0,
    ])
    let bias = @cpu.CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0])
    @model.Linear::{ weight, bias }.forward(input).values()
  })
}

///|
fn run_add() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let shape = @shape.Shape::new([4])
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    let input_binding = graph.input("input", shape)
    let input = input_binding.tensor()
    let bias = graph.constant(shape, [10.0, 20.0, 30.0, 40.0])
    let output = input.add(bias)
    let session = graph.compile_single(input_binding, "output", output)
    session.run([1.0, 2.0, 3.0, 4.0])
  })
}

///|
fn run_concat() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let lhs = graph.input("lhs", @shape.Shape::new([2, 2]))
    let rhs = graph.input("rhs", @shape.Shape::new([2, 1]))
    let output = lhs.tensor().concat(rhs.tensor(), 1)
    let program = graph.compile_program_named([lhs, rhs], [
      graph.output("output", output),
    ])
    defer program.destroy()
    let results = program.run_named([
      @webnn.WebNNNamedValues::new("lhs", [1.0, 2.0, 3.0, 4.0]),
      @webnn.WebNNNamedValues::new("rhs", [10.0, 20.0]),
    ])
    results[0].values()
  })
}

///|
fn run_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let input_shape = @shape.Shape::new([1, 2])
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    let input_binding = graph.input("input", input_shape)
    let input = input_binding.tensor()
    let weight = graph.constant(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
    let bias = graph.constant(@shape.Shape::new([2]), [1.0, 1.0])
    let linear = @model.Linear::{ weight, bias }
    let output = linear.forward(input)
    let session = graph.compile_single(input_binding, "output", output)
    session.run([1.0, 2.0])
  })
}

///|
fn run_litert_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let input_shape = @shape.Shape::new([1, 2])
    let weight_shape = @shape.Shape::new([2, 2])
    let output_shape = @shape.Shape::new([1, 2])
    let model = @litert.LiteRtModel::new(
      [
        @litert.LiteRtValue::input("input", input_shape),
        @litert.LiteRtValue::constant("weight", weight_shape, [
          1.0, 2.0, 3.0, 4.0,
        ]),
        @litert.LiteRtValue::constant("bias", @shape.Shape::new([2]), [1.0, 1.0]),
        @litert.LiteRtValue::intermediate("matmul", output_shape),
        @litert.LiteRtValue::intermediate("output", output_shape),
      ],
      [
        @litert.LiteRtNode::Matmul("input", "weight", "matmul"),
        @litert.LiteRtNode::Add("matmul", "bias", "output"),
      ],
      ["output"],
    )
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let lowered = builder.lower_litert(model)
    let input_bindings = lowered.inputs()
    let outputs = lowered.outputs()
    let program = builder.compile_program_single(
      input_bindings[0],
      outputs[0].name(),
      outputs[0].tensor(),
    )
    defer program.destroy()
    program.run([1.0, 2.0])
  })
}

///|
extern "js" fn tflite_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAGAAAACwBAAABAAAACAAAAAQABAAEAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAALgAAAC8AAAAwAAAAAMAAAAYAAAASAAAAHgAAAAMABAABAAAAAgADAAMAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAEAAAAAIAAAABAAAAAgAAAAYAAAB3ZWlnaHQAAAwAEAAEAAAACAAMAAwAAAAMAAAAAAAAABAAAAACAAAAAQAAAAIAAAAGAAAAb3V0cHV0AAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKAAwAAAAEAAgAAAAMAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAIAAAAMAAAAFAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAAAgAAAAAAOBAAAAQQQ=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_two_input_two_output_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAAWAAAAAAAAACcAQAAAgAAABQAAAAwAAAADAAUAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAEwAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACwAAAAuAAAAMAAAAAEAAAAHAAAADwAAABcAAAAfAAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAIAAAAAAAAAAQAAAAIAAAACAAAAAwAAAAIAAAAUAAAAQAAAAAoAEAAAAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAoAEAAEAAgADAAAAAwAAAABAAAACAAAAAwAAAABAAAAAAAAAAEAAAADAAAAAQAAAAwAAAAGAAgAAAAAAAgAAAAAAAAA"),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_sub_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAcAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAKQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACIAAAAkAAAAJQAAAADAAAAGAAAADgAAABYAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAAAgAAAAAAAAABAAAAAQAAAAIAAAABAAAAEAAAAAoAEAAAAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAMAAAABgAIAAAAAAAIAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_add_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAGAAAACwBAAABAAAACAAAAAQABAAEAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAALgAAAC8AAAAwAAAAAMAAAAYAAAASAAAAHgAAAAMABAABAAAAAgADAAMAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAEAAAAAIAAAABAAAAAgAAAAYAAAB3ZWlnaHQAAAwAEAAEAAAACAAMAAwAAAAMAAAAAAAAABAAAAACAAAAAQAAAAIAAAAGAAAAb3V0cHV0AAABAAAAAAAAAAEAAAACAAAAAQAAAHAAAAAKAAwAAAAEAAgAAAAMAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAIAAAAMAAAAFAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAAAgAAAAAAOBAAAAQQQ4AFAAAAAQACAAMABAAAAAQAAAAEAAAABgAAAALAAAAIAAAAAIAAAAAAAAAAQAAAAEAAAACAAAABgAIAAQAAAAIAAAAAwAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_fully_connected_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAAHABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAAJAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAOwAAADwAAAA9AAAAAQAAAAcAAAATAAAAHwAAACoAAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAEAAAAAIAAAABAAAAAgAAAAUAAABpbnB1dAAMABAABAAAAAgADAAAAA4AAAAMAAAAAQAAABAAAAACAAAAAgAAAAIAAAAHAAAAd2VpZ2h0cwAMABAABAAAAAgADAAMAAAADAAAAAIAAAAMAAAAAQAAAAIAAAAEAAAAYmlhcwAADAAQAAQAAAAIAAwAAAAOAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAAAQAAAACgAMAAAABAAIAAAADAAAAAgAAAAUAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAAwAAABAAAAAYAAAAOAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAABAAAAAAAIA/AABAQAAAAEAAAIBABgAIAAQAAAAIAAAABAAAAAgAAAAAAIA/AACAPw=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_fully_connected_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAAHABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAAJAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAOwAAADwAAAA9AAAAAQAAAAcAAAATAAAAHwAAACoAAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAEAAAAAIAAAABAAAAAgAAAAUAAABpbnB1dAAMABAABAAAAAgADAAAAA4AAAAMAAAAAQAAABAAAAACAAAAAgAAAAIAAAAHAAAAd2VpZ2h0cwAMABAABAAAAAgADAAMAAAADAAAAAIAAAAMAAAAAQAAAAIAAAAEAAAAYmlhcwAADAAQAAQAAAAIAAwAAAAOAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAACcAAAACgAMAAAABAAIAAAADAAAAAgAAAAUAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAAwAAABAAAAAYAAAAOAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAABAAAAAAAIA/AABAQAAAAEAAAIBABgAIAAQAAAAIAAAABAAAAAgAAAAAAIA/AACAPw4AFAAAAAQACAAMABAAAAAQAAAAEAAAABwAAAAIAAAAJAAAAAMAAAAAAAAAAQAAAAIAAAABAAAAAwAAAAYACAAEAAAACAAAAAMAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_conv2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAALABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAADAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAAQBAAAIAQAADAEAAAQAAAAcAAAAVAAAAIwAAAC4AAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAGAAAAAQAAAABAAAAAwAAAAMAAAABAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAGAAAAAQAAAABAAAAAgAAAAIAAAABAAAABgAAAGZpbHRlcgAADAAQAAQAAAAIAAwADAAAAAwAAAACAAAADAAAAAEAAAABAAAABAAAAGJpYXMAAAwAEAAEAAAACAAMAAAADgAAAAwAAAAAAAAAGAAAAAQAAAABAAAAAgAAAAIAAAABAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAAAUAAAADgAUAAAABAAIAAwAEAAAABAAAAAQAAAAHAAAAAEAAAAoAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAACgAQAAQACAAMAAAADAAAAAEAAAABAAAAAQAAAAMAAAAQAAAAGAAAADgAAAAEAAQABAAAAAYACAAEAAAACAAAAAQAAAAQAAAAAACAPwAAAAAAAAAAAACAvwYACAAEAAAACAAAAAQAAAAEAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_reshape_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAsAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFgAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACcAAAAoAAAAKQAAAADAAAAGAAAAEAAAABkAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAgAAAAIAAAAMABQABAAIAAwAAAAMAAAAEAAAAAIAAAABAAAAAAAAAAEAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAABAAAAAEAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAgAAAAEAAAAAQAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_transpose_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAsAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAJwAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACcAAAAoAAAAKQAAAADAAAAGAAAAEAAAABkAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAgAAAAMAAAAMABQABAAIAAwAAAAMAAAAEAAAAAIAAAABAAAAAAAAAAEAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAwAAAAIAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAgAAAABAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_depthwise_conv2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAC4AQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAABAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAADkAAAA6AAAAOwAAAAEAAAAHAAAAEwAAAB8AAAAoAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAADAAAAAwAAAAIAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAABAAAAAAAAAAQAAAABAAAAAgAAAAIAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAgAAAAAAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAACAAAAAgAAAAIAAAABAAAAAAAAAAEAAAADAAAAAQAAABQAAAAOABgABAAIAAwAAAAUAAAAEAAAAAAAAAAQAAAAHAAAAAAAAAAwAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAEgAgAAQACAAMABAAFAAYABwAAAAUAAAAAQAAAAEAAAABAAAAAQAAAAAAAAABAAAAAQAAAAMAAAAUAAAAIAAAAFAAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAACAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEAAAOBAAAAAQQYACAAEAAAACAAAAAQAAAAIAAAAAACAPwAAIME="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_average_pool2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAABEAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAAAAhAAAAIgAAAACAAAAFAAAAEQAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAQAAAABAAAAAwAAAAMAAAABAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAAEAAAAAQAAAAIAAAACAAAAAQAAAAEAAAAAAAAAAQAAAAEAAAABAAAAFAAAAA4AGAAEAAgADAAAABQAAAAQAAAAAAAAABAAAAAUAAAAAAAAACQAAAABAAAAAAAAAAEAAAABAAAAEAAcAAQACAAMABAAFAAYABAAAAABAAAAAQAAAAEAAAACAAAAAgAAAAAAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_max_pool2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAABEAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAEQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAAAAhAAAAIgAAAACAAAAFAAAAEQAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAQAAAABAAAAAwAAAAMAAAABAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAAEAAAAAQAAAAIAAAACAAAAAQAAAAEAAAAAAAAAAQAAAAEAAAABAAAAFAAAAA4AGAAEAAgADAAAABQAAAAQAAAAAAAAABAAAAAUAAAAAAAAACQAAAABAAAAAAAAAAEAAAABAAAAEAAcAAQACAAMABAAFAAYABAAAAABAAAAAQAAAAEAAAACAAAAAgAAAAAAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_logistic_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAADgAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAACAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_tanh_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAHAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAACAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_uint8_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAAAwAAAAAAAAAAAAAAIAAAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAgAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAADAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAgAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAADAAAAAAAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAgAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAACChA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_int8_per_axis_runtime_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAACQAAAAAAAAAAAAAAzAEAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAAAAAAAAAAAkAQAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAAACBAAAEAAQAAAAAAAEAAgAAAAMABAAAAAMAAAAGAAAAAEAAAACAAAAAAAAPwAAgD4AAAAAAgAAAAEAAAAAAAAA/v////////8QABAAAAAAAAQACAAAAAwAEAAAAAwAAAAUAAAAAQAAAAIAAAAAAAA/AACAPgIAAAABAAAAAAAAAP7/////////"),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_relu_n1_to_1_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAABAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAAEAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAABAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAAEAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
fn run_tflite(
  bytes : Bytes,
  input_values : Array[Float],
) -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let model = @litert.TfliteModel::parse(bytes)
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let program = builder.compile_litert_program(model)
    defer program.destroy()
    program.run(input_values)
  })
}

///|
fn run_tflite_add() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_add_fixture(), [1.0, 2.0])
}

///|
fn run_tflite_add_relu6() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_add_relu6_fixture(), [1.0, 2.0])
}

///|
fn run_tflite_fully_connected() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_fully_connected_fixture(), [1.0, 2.0])
}

///|
fn run_tflite_fully_connected_relu6() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_fully_connected_relu6_fixture(), [1.0, 2.0])
}

///|
fn run_tflite_conv2d() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_conv2d_fixture(), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
}

///|
fn run_tflite_reshape() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_reshape_fixture(), [1.0, 2.0, 3.0, 4.0])
}

///|
fn run_tflite_transpose() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_transpose_fixture(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

///|
fn run_tflite_depthwise_conv2d() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_depthwise_conv2d_fixture(), [
    1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0,
    8.0, 80.0, 9.0, 90.0,
  ])
}

///|
fn run_tflite_average_pool2d() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_average_pool2d_fixture(), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
}

///|
fn run_tflite_max_pool2d() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_max_pool2d_fixture(), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
}

///|
fn run_tflite_logistic() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_logistic_fixture(), [0.0, 1.0])
}

///|
fn run_tflite_tanh() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_tanh_fixture(), [0.0, 1.0])
}

///|
fn run_tflite_quantized_add() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_uint8_add_fixture(), [1.0, 2.0])
}

///|
/// Demonstrates metadata-driven raw UINT8 I/O around the float32 backend.
fn run_tflite_quantized_add_raw() -> @js.Promise[Array[Int]] {
  @js.from_async(async fn() {
    let runtime = @litert.TfliteModel::parse_runtime(tflite_uint8_add_fixture())
    let runtime_inputs = runtime.inputs()
    let runtime_outputs = runtime.outputs()
    let input = runtime_inputs[0].dequantize([130, 132])
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let program = builder.compile_litert_program(runtime.model())
    defer program.destroy()
    runtime_outputs[0].requantize(program.run(input))
  })
}

///|
/// Demonstrates parser-retained per-axis INT8 I/O around the float32 backend.
fn run_tflite_per_axis_int8_raw() -> @js.Promise[Array[Int]] {
  @js.from_async(async fn() {
    let runtime = @litert.TfliteModel::parse_runtime(
      tflite_int8_per_axis_runtime_add_fixture(),
    )
    let runtime_inputs = runtime.inputs()
    let runtime_outputs = runtime.outputs()
    let input = runtime_inputs[0].dequantize([3, 2])
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let program = builder.compile_litert_program(runtime.model())
    defer program.destroy()
    runtime_outputs[0].requantize(program.run(input))
  })
}

///|
let tflite_runner_cache : @webnn.TfliteRunnerCache = @webnn.TfliteRunnerCache::new()

///|
extern "js" fn tflite_runner_cache_report(
  first : Array[Int],
  second : Array[Int],
  cache_size : Int,
) -> @core.Any =
  #| (first, second, cacheSize) => ({ first, second, cacheSize })

///|
fn quantized_tflite_runner_output(
  output : @webnn.TfliteRunnerOutput,
) -> Array[Int] raise @tensor.TensorError {
  match output.values() {
    @webnn.TfliteRunnerValues::Quantized(values) => values
    @webnn.TfliteRunnerValues::Float32(_) =>
      raise @tensor.TensorError::new(
        "expected a quantized TFLite runner output",
      )
  }
}

///|
/// Exercises arbitrary-FlatBuffer compilation, metadata-driven raw INT8 I/O,
/// and SHA-256 keyed program reuse through the public runner API.
fn run_tflite_runner_cache_per_axis_int8_raw() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let _ = tflite_runner_cache.clear()
    let bytes = tflite_int8_per_axis_runtime_add_fixture()
    let first_outputs = tflite_runner_cache.run(
      bytes,
      @compat.DevicePreference::Npu,
      [@webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2])],
    )
    let second_outputs = tflite_runner_cache.run(
      bytes,
      @compat.DevicePreference::Npu,
      [@webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2])],
    )
    tflite_runner_cache_report(
      quantized_tflite_runner_output(first_outputs[0]),
      quantized_tflite_runner_output(second_outputs[0]),
      tflite_runner_cache.length(),
    )
  })
}

///|
/// Exercises the public WebNNRuntime entry point and its owned TFLite cache.
fn run_tflite_runtime_cache_per_axis_int8_raw() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let runtime = @webnn.WebNNRuntime::new(@compat.DevicePreference::Npu)
    defer runtime.clear() |> ignore
    let bytes = tflite_int8_per_axis_runtime_add_fixture()
    let first_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2]),
    ])
    let second_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2]),
    ])
    tflite_runner_cache_report(
      quantized_tflite_runner_output(first_outputs[0]),
      quantized_tflite_runner_output(second_outputs[0]),
      runtime.cache_size(),
    )
  })
}

///|
/// Exercises the CPU TFLite runner with the same raw per-axis INT8 boundary.
fn run_tflite_cpu_per_axis_int8_raw() -> @js.Promise[Array[Int]] {
  @js.from_async(async fn() {
    let _ = @js.Promise::resolve(()).wait()
    let runner = @webnn.TfliteCpuRunner::compile(
      tflite_int8_per_axis_runtime_add_fixture(),
    )
    let outputs = runner.run([
      @webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2]),
    ])
    quantized_tflite_runner_output(outputs[0])
  })
}

///|
extern "js" fn tflite_runtime_cpu_fallback_report(
  output : Array[Int],
  misses : Int,
  entries : Int,
  cpu_fallbacks : Int,
) -> @core.Any =
  #| (output, misses, entries, cpuFallbacks) => ({
  #|   output, misses, entries, cpuFallbacks,
  #| })

///|
/// Exercise the explicit CPU fallback policy. Tests temporarily hide
/// navigator.ml so the initial WebNN compile fails before this runner executes.
fn run_tflite_runtime_cpu_fallback_per_axis_int8_raw() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let runtime = @webnn.WebNNRuntime::new_with_cpu_fallback(
      @compat.DevicePreference::Npu,
      1,
      2,
    )
    defer runtime.clear() |> ignore
    let outputs = runtime.run_tflite(
      tflite_int8_per_axis_runtime_add_fixture(),
      [@webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2])],
    )
    let metrics = runtime.metrics()
    tflite_runtime_cpu_fallback_report(
      quantized_tflite_runner_output(outputs[0]),
      metrics.misses(),
      metrics.entries(),
      metrics.cpu_fallbacks(),
    )
  })
}

///|
extern "js" fn tflite_runtime_policy_report(
  hits : Int,
  misses : Int,
  evictions : Int,
  entries : Int,
  cpu_fallbacks : Int,
) -> @core.Any =
  #| (hits, misses, evictions, entries, cpuFallbacks) => ({
  #|   hits, misses, evictions, entries, cpuFallbacks,
  #| })

///|
/// Verifies capacity-two LRU behavior: touching A preserves A while B is
/// evicted when C is compiled; a later B run is therefore a second miss.
fn probe_tflite_runtime_policy() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let runtime = @webnn.WebNNRuntime::new_with_options(
      @compat.DevicePreference::Npu,
      1,
      2,
      @webnn.WebNNRuntimeFallback::Disabled,
    )
    defer runtime.clear() |> ignore
    let _ = runtime.run_tflite(tflite_int8_per_axis_runtime_add_fixture(), [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2]),
    ])
    let _ = runtime.run_tflite(tflite_uint8_add_fixture(), [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [130, 132]),
    ])
    let _ = runtime.run_tflite(tflite_int8_per_axis_runtime_add_fixture(), [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2]),
    ])
    let _ = runtime.run_tflite(tflite_add_fixture(), [
      @webnn.TfliteRunnerInput::float32("input", [1.0, 2.0]),
    ])
    let _ = runtime.run_tflite(tflite_uint8_add_fixture(), [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [130, 132]),
    ])
    let metrics = runtime.metrics()
    tflite_runtime_policy_report(
      metrics.hits(),
      metrics.misses(),
      metrics.evictions(),
      metrics.entries(),
      metrics.cpu_fallbacks(),
    )
  })
}

///|
extern "js" fn tflite_runtime_cpu_fallback_cache_report(
  first : Array[Int],
  second : Array[Int],
  cpu_fallbacks : Int,
  cpu_cache_hits : Int,
  cpu_cache_misses : Int,
  cpu_cache_entries : Int,
  cpu_cache_resident_bytes : Int,
) -> @core.Any =
  #| (first, second, cpuFallbacks, cpuCacheHits, cpuCacheMisses, cpuCacheEntries, cpuCacheResidentBytes) => ({
  #|   first,
  #|   second,
  #|   cpuFallbacks,
  #|   cpuCacheHits,
  #|   cpuCacheMisses,
  #|   cpuCacheEntries,
  #|   cpuCacheResidentBytes,
  #| })

///|
/// Runs the same failed-WebNN request twice to verify parsed CPU fallback
/// models are reused instead of parsing and lowering again.
fn probe_tflite_runtime_cpu_fallback_cache() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let runtime = @webnn.WebNNRuntime::new_with_cpu_fallback(
      @compat.DevicePreference::Npu,
      1,
      2,
    )
    defer runtime.clear() |> ignore
    let bytes = tflite_int8_per_axis_runtime_add_fixture()
    let inputs = [@webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2])]
    let first = runtime.run_tflite(bytes, inputs)
    let second = runtime.run_tflite(bytes, inputs)
    let metrics = runtime.metrics()
    tflite_runtime_cpu_fallback_cache_report(
      quantized_tflite_runner_output(first[0]),
      quantized_tflite_runner_output(second[0]),
      metrics.cpu_fallbacks(),
      metrics.cpu_cache_hits(),
      metrics.cpu_cache_misses(),
      metrics.cpu_cache_entries(),
      metrics.cpu_cache_resident_bytes(),
    )
  })
}

///|
extern "js" fn tflite_runtime_byte_budget_report(
  hits : Int,
  misses : Int,
  evictions : Int,
  entries : Int,
  resident_bytes : Int,
  byte_capacity : Int,
) -> @core.Any =
  #| (hits, misses, evictions, entries, residentBytes, byteCapacity) => ({
  #|   hits,
  #|   misses,
  #|   evictions,
  #|   entries,
  #|   residentBytes,
  #|   byteCapacity,
  #| })

///|
/// Verifies LRU eviction by source-FlatBuffer byte budget independently of the
/// entry-count budget.
fn probe_tflite_runtime_byte_budget() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let float_bytes = tflite_add_fixture()
    let uint8_bytes = tflite_uint8_add_fixture()
    let byte_capacity = if float_bytes.length() > uint8_bytes.length() {
      float_bytes.length()
    } else {
      uint8_bytes.length()
    }
    let runtime = @webnn.WebNNRuntime::new_with_cache_limits(
      @compat.DevicePreference::Npu,
      1,
      4,
      byte_capacity,
      1,
      byte_capacity,
      @webnn.WebNNRuntimeFallback::Disabled,
    )
    defer runtime.clear() |> ignore
    runtime.run_tflite(float_bytes, [
      @webnn.TfliteRunnerInput::float32("input", [1.0, 2.0]),
    ])
    |> ignore
    runtime.run_tflite(uint8_bytes, [
      @webnn.TfliteRunnerInput::quantized("tensor_0", [130, 132]),
    ])
    |> ignore
    runtime.run_tflite(float_bytes, [
      @webnn.TfliteRunnerInput::float32("input", [1.0, 2.0]),
    ])
    |> ignore
    let metrics = runtime.metrics()
    tflite_runtime_byte_budget_report(
      metrics.hits(),
      metrics.misses(),
      metrics.evictions(),
      metrics.entries(),
      metrics.resident_bytes(),
      metrics.byte_capacity(),
    )
  })
}

///|
extern "js" fn fetch_tflite_fixture(url : String) -> @js.Promise[Bytes] =
  #| async (url) => {
  #|   const response = await fetch(url);
  #|   if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  #|   return new Uint8Array(await response.arrayBuffer());
  #| }

///|
extern "js" fn make_micro_speech_fixture_report(
  input_name : String,
  input_shape : Array[Int],
  output_name : String,
  output_shape : Array[Int],
  first : Array[Int],
  second : Array[Int],
  cache_size : Int,
) -> @core.Any =
  #| (inputName, inputShape, outputName, outputShape, first, second, cacheSize) => ({
  #|   inputName,
  #|   inputShape,
  #|   outputName,
  #|   outputShape,
  #|   first,
  #|   second,
  #|   cacheSize,
  #| })

///|
extern "js" fn make_micro_speech_fixture_error_report(
  error : String,
) -> @core.Any =
  #| (error) => ({ error })

///|
/// Run TensorFlow Lite Micro's quantized Micro Speech model from its pinned
/// fixture. Zero-valued raw INT8 input makes the test deterministic while
/// exercising a real CONV_2D → FULLY_CONNECTED → SOFTMAX model.
fn run_micro_speech_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/tflite/micro_speech_quantized.tflite",
    ).wait()
    let model = try @litert.TfliteModel::parse_runtime(bytes) catch {
      @litert.LiteRtError::LiteRtError(message) =>
        return make_micro_speech_fixture_error_report(message)
    } noraise {
      model => model
    }
    let input = model.inputs()[0]
    let output = model.outputs()[0]
    let runtime = @webnn.WebNNRuntime::new(@compat.DevicePreference::Npu)
    defer runtime.clear() |> ignore
    let raw_input = Array::make(input.shape().element_count(), 0)
    let first_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    let second_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    make_micro_speech_fixture_report(
      input.name(),
      input.shape().dimensions(),
      output.name(),
      output.shape().dimensions(),
      quantized_tflite_runner_output(first_outputs[0]),
      quantized_tflite_runner_output(second_outputs[0]),
      runtime.cache_size(),
    )
  })
}

///|
/// Run TensorFlow Lite Micro's int8 person-detection model from its pinned
/// fixture. A zero-valued image gives deterministic raw quantized I/O while
/// exercising a substantially deeper MobileNet-style graph.
fn run_person_detection_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture("/fixtures/tflite/person_detect.tflite").wait()
    let model = try @litert.TfliteModel::parse_runtime(bytes) catch {
      @litert.LiteRtError::LiteRtError(message) =>
        return make_micro_speech_fixture_error_report(message)
    } noraise {
      model => model
    }
    let input = model.inputs()[0]
    let output = model.outputs()[0]
    let runtime = @webnn.WebNNRuntime::new(@compat.DevicePreference::Npu)
    defer runtime.clear() |> ignore
    let raw_input = Array::make(input.shape().element_count(), 0)
    let first_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    let second_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    make_micro_speech_fixture_report(
      input.name(),
      input.shape().dimensions(),
      output.name(),
      output.shape().dimensions(),
      quantized_tflite_runner_output(first_outputs[0]),
      quantized_tflite_runner_output(second_outputs[0]),
      runtime.cache_size(),
    )
  })
}

///|
extern "js" fn make_mobilenet_v2_fixture_report(
  input_shape : Array[Int],
  output_shape : Array[Int],
) -> @core.Any =
  #| (inputShape, outputShape) => ({ inputShape, outputShape })

///|
extern "js" fn performance_now() -> Double =
  #| () => performance.now()

///|
extern "js" fn make_mobilenet_v2_benchmark_report(
  warmup : Int,
  iterations : Int,
  cold_samples_ms : Array[Double],
  hit_samples_ms : Array[Double],
  prepared_hit_samples_ms : Array[Double],
  output : Array[Int],
  cache_size : Int,
) -> @core.Any =
  #| (warmup, iterations, coldSamplesMs, hitSamplesMs, preparedHitSamplesMs, output, cacheSize) => ({
  #|   warmup, iterations, coldSamplesMs, hitSamplesMs, preparedHitSamplesMs, output, cacheSize,
  #| })

///|
/// Probe a standard MobileNet V2 quantized model to select the next parser
/// operator from a real model corpus before adding synthetic coverage.
fn probe_mobilenet_v2_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/tflite/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
    ).wait()
    let model = try @litert.TfliteModel::parse_runtime(bytes) catch {
      @litert.LiteRtError::LiteRtError(message) =>
        return make_micro_speech_fixture_error_report(message)
    } noraise {
      model => model
    }
    make_mobilenet_v2_fixture_report(
      model.inputs()[0].shape().dimensions(),
      model.outputs()[0].shape().dimensions(),
    )
  })
}

///|
/// Execute MobileNet V2 twice to cover actual lowering, compile, raw UINT8 I/O,
/// and the cache-hit path for the large real-model corpus fixture.
fn run_mobilenet_v2_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/tflite/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
    ).wait()
    let model = try @litert.TfliteModel::parse_runtime(bytes) catch {
      @litert.LiteRtError::LiteRtError(message) =>
        return make_micro_speech_fixture_error_report(message)
    } noraise {
      model => model
    }
    let input = model.inputs()[0]
    let output = model.outputs()[0]
    let runtime = @webnn.WebNNRuntime::new(@compat.DevicePreference::Npu)
    defer runtime.clear() |> ignore
    let raw_input = Array::make(input.shape().element_count(), 0)
    let first_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    let second_outputs = runtime.run_tflite(bytes, [
      @webnn.TfliteRunnerInput::quantized(input.name(), raw_input),
    ])
    make_micro_speech_fixture_report(
      input.name(),
      input.shape().dimensions(),
      output.name(),
      output.shape().dimensions(),
      quantized_tflite_runner_output(first_outputs[0]),
      quantized_tflite_runner_output(second_outputs[0]),
      runtime.cache_size(),
    )
  })
}

///|
fn benchmark_mobilenet_v2_fixture(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    if warmup < 0 {
      raise @tensor.TensorError::new(
        "MobileNet V2 benchmark warmup must be non-negative",
      )
    }
    if iterations <= 0 {
      raise @tensor.TensorError::new(
        "MobileNet V2 benchmark iterations must be positive",
      )
    }
    let bytes = fetch_tflite_fixture(
      "/fixtures/tflite/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
    ).wait()
    let model = @litert.TfliteModel::parse_runtime(bytes)
    let input = model.inputs()[0]
    let raw_input = Array::make(input.shape().element_count(), 0)
    let inputs = [@webnn.TfliteRunnerInput::quantized(input.name(), raw_input)]
    let runtime = @webnn.WebNNRuntime::new(@compat.DevicePreference::Npu)
    defer runtime.clear() |> ignore
    let cold_samples_ms : Array[Double] = []
    let mut output : Array[Int] = []
    for _ in 0..<iterations {
      let _ = runtime.clear()
      let start = performance_now()
      output = quantized_tflite_runner_output(
        runtime.run_tflite(bytes, inputs)[0],
      )
      cold_samples_ms.push(performance_now() - start)
    }
    let _ = runtime.clear()
    output = quantized_tflite_runner_output(
      runtime.run_tflite(bytes, inputs)[0],
    )
    for _ in 0..<warmup {
      let _ = runtime.run_tflite(bytes, inputs)
    }
    let hit_samples_ms : Array[Double] = []
    for _ in 0..<iterations {
      let start = performance_now()
      output = quantized_tflite_runner_output(
        runtime.run_tflite(bytes, inputs)[0],
      )
      hit_samples_ms.push(performance_now() - start)
    }
    let artifact = @webnn.TfliteModelArtifact::from_bytes(bytes)
    let _ = runtime.clear()
    output = quantized_tflite_runner_output(
      runtime.run_prepared_tflite(artifact, inputs)[0],
    )
    for _ in 0..<warmup {
      let _ = runtime.run_prepared_tflite(artifact, inputs)
    }
    let prepared_hit_samples_ms : Array[Double] = []
    for _ in 0..<iterations {
      let start = performance_now()
      output = quantized_tflite_runner_output(
        runtime.run_prepared_tflite(artifact, inputs)[0],
      )
      prepared_hit_samples_ms.push(performance_now() - start)
    }
    make_mobilenet_v2_benchmark_report(
      warmup,
      iterations,
      cold_samples_ms,
      hit_samples_ms,
      prepared_hit_samples_ms,
      output,
      runtime.cache_size(),
    )
  })
}

///|
/// Verify that unsupported real-model operators fail with their TFLite name.
fn probe_audio_preprocessor_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/tflite/audio_preprocessor_int8.tflite",
    ).wait()
    try {
      @litert.TfliteModel::parse_runtime(bytes) |> ignore
      make_micro_speech_fixture_error_report(
        "Audio Preprocessor fixture unexpectedly parsed",
      )
    } catch {
      @litert.LiteRtError::LiteRtError(message) =>
        make_micro_speech_fixture_error_report(message)
    } noraise {
      _ =>
        make_micro_speech_fixture_error_report(
          "Audio Preprocessor fixture unexpectedly parsed",
        )
    }
  })
}

///|
/// Verifies that concurrent cache misses/hits keep raw quantized values isolated.
fn probe_tflite_runner_cache_concurrency() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let cache = @webnn.TfliteRunnerCache::new()
    defer ignore(cache.clear())
    let run = fn(values : Array[Int]) -> @js.Promise[Array[Int]] {
      @js.from_async(async fn() {
        let outputs = cache.run(
          tflite_int8_per_axis_runtime_add_fixture(),
          @compat.DevicePreference::Npu,
          [@webnn.TfliteRunnerInput::quantized("tensor_0", values)],
        )
        if outputs.length() != 1 {
          raise @tensor.TensorError::new(
            "expected exactly one TFLite runner output",
          )
        }
        quantized_tflite_runner_output(outputs[0])
      })
    }
    let results = run_int_program_pair(run, [3, 2], [4, 2]).wait()
    make_tflite_runner_cache_concurrency_report(results, cache.length())
  })
}

///|
/// Uses the input/output names embedded in a standard TFLite ADD FlatBuffer.
fn run_tflite_runner_named_add() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let runner = @webnn.TfliteRunner::compile(
      tflite_add_fixture(),
      @compat.DevicePreference::Npu,
    )
    defer runner.destroy()
    let outputs = runner.run([
      @webnn.TfliteRunnerInput::float32("input", [1.0, 2.0]),
    ])
    if outputs.length() != 1 {
      raise @tensor.TensorError::new(
        "expected exactly one TFLite runner output",
      )
    }
    match outputs[0].values() {
      @webnn.TfliteRunnerValues::Float32(values) => values
      @webnn.TfliteRunnerValues::Quantized(_) =>
        raise @tensor.TensorError::new(
          "expected a float32 TFLite runner output",
        )
    }
  })
}

///|
extern "js" fn tflite_runner_multi_io_report(
  names : Array[String],
  outputs : Array[Array[Float]],
) -> @core.Any =
  #| (names, outputs) => ({
  #|   names,
  #|   outputs: Object.fromEntries(names.map((name, index) => [name, outputs[index]])),
  #| })

///|
/// Runs a TFLite graph with two named inputs and two declared outputs.
fn run_tflite_runner_two_input_two_output() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let runner = @webnn.TfliteRunner::compile(
      tflite_two_input_two_output_fixture(),
      @compat.DevicePreference::Npu,
    )
    defer runner.destroy()
    let runner_outputs = runner.run([
      @webnn.TfliteRunnerInput::float32("tensor_1", [10.0, 20.0]),
      @webnn.TfliteRunnerInput::float32("tensor_0", [1.0, 2.0]),
    ])
    let names : Array[String] = []
    let outputs : Array[Array[Float]] = []
    for output in runner_outputs {
      names.push(output.name())
      match output.values() {
        @webnn.TfliteRunnerValues::Float32(values) => outputs.push(values)
        @webnn.TfliteRunnerValues::Quantized(_) =>
          raise @tensor.TensorError::new(
            "expected a float32 TFLite runner output",
          )
      }
    }
    tflite_runner_multi_io_report(names, outputs)
  })
}

///|
fn run_tflite_sub() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let runner = @webnn.TfliteRunner::compile(
      tflite_sub_fixture(),
      @compat.DevicePreference::Npu,
    )
    defer runner.destroy()
    let outputs = runner.run([
      @webnn.TfliteRunnerInput::float32("tensor_0", [3.0, 20.0]),
      @webnn.TfliteRunnerInput::float32("tensor_1", [1.0, 10.0]),
    ])
    if outputs.length() != 1 {
      raise @tensor.TensorError::new(
        "expected exactly one TFLite runner output",
      )
    }
    match outputs[0].values() {
      @webnn.TfliteRunnerValues::Float32(values) => values
      @webnn.TfliteRunnerValues::Quantized(_) =>
        raise @tensor.TensorError::new(
          "expected a float32 TFLite runner output",
        )
    }
  })
}

///|
fn run_tflite_relu_n1_to_1() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_relu_n1_to_1_fixture(), [-2.0, -0.5, 0.5, 3.0])
}

///|
fn run_tflite_relu6() -> @js.Promise[Array[Float]] {
  run_tflite(tflite_relu6_fixture(), [-2.0, 0.5, 5.0, 8.0])
}

///|
fn benchmark_litert_linear(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.litert_linear(warmup, iterations) })
}

///|
fn benchmark_tflite_fully_connected(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.tflite_fully_connected(warmup, iterations)
  })
}

///|
fn benchmark_tflite_sub(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.tflite_sub(warmup, iterations) })
}

///|
fn benchmark_tflite_depthwise_conv2d(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.tflite_depthwise_conv2d(warmup, iterations)
  })
}

///|
fn benchmark_tflite_max_pool2d(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.tflite_max_pool2d(warmup, iterations) })
}

///|
fn benchmark_tflite_per_axis_int8_raw(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.tflite_per_axis_int8_raw(warmup, iterations)
  })
}

///|
fn benchmark_tflite_runner_cache_per_axis_int8_raw(
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.tflite_runner_cache_per_axis_int8_raw(warmup, iterations)
  })
}

///|
fn run_transform_pipeline() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let input_shape = @shape.Shape::new([2, 2])
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    let input_binding = graph.input("input", input_shape)
    let input = input_binding.tensor()
    let multiplier = graph.constant(input_shape, [2.0, 3.0, 4.0, 5.0])
    let divisor = graph.constant(input_shape, [2.0, 2.0, 2.0, 2.0])
    let normalization_shape = @shape.Shape::new([1])
    let normalization_scale = graph.constant(normalization_shape, [1.0])
    let normalization_bias = graph.constant(normalization_shape, [0.25])
    let output = input
      .gather([1, 0], @shape.Shape::new([2]), 0)
      .mul(multiplier)
      .div(divisor)
      .reduce_mean([1], true)
      .slice([1, 0], [1, 1])
      .layer_normalization(normalization_scale, normalization_bias, [1], 1.0e-5)
      .gelu()
      .reshape(@shape.Shape::new([1, 1]))
      .transpose([1, 0])
    let session = graph.compile_single(input_binding, "output", output)
    session.run([1.0, 2.0, 3.0, 4.0])
  })
}

///|
fn main {
  expose_api(
    @compat.compatibility_mode, supported_operators, probe_webnn_capabilities, run_cpu_linear,
    run_add, run_concat, run_linear, run_litert_linear, run_tflite_add, run_tflite_add_relu6,
    run_tflite_fully_connected, run_tflite_fully_connected_relu6, run_tflite_conv2d,
    run_tflite_reshape, run_tflite_transpose, run_tflite_depthwise_conv2d, run_tflite_average_pool2d,
    run_tflite_max_pool2d, run_tflite_logistic, run_tflite_tanh, run_tflite_quantized_add,
    run_tflite_quantized_add_raw, run_tflite_per_axis_int8_raw, run_tflite_runner_cache_per_axis_int8_raw,
    run_tflite_runtime_cache_per_axis_int8_raw, run_tflite_cpu_per_axis_int8_raw,
    run_tflite_runtime_cpu_fallback_per_axis_int8_raw, probe_tflite_runtime_policy,
    probe_tflite_runtime_cpu_fallback_cache, probe_tflite_runtime_byte_budget, run_micro_speech_fixture,
    run_person_detection_fixture, probe_mobilenet_v2_fixture, run_mobilenet_v2_fixture,
    benchmark_mobilenet_v2_fixture, probe_audio_preprocessor_fixture, probe_tflite_runner_cache_concurrency,
    probe_tflite_runner_cache_clear_during_compile, run_tflite_runner_named_add,
    run_tflite_runner_two_input_two_output, run_tflite_sub, run_tflite_relu_n1_to_1,
    run_tflite_relu6, benchmark_litert_linear, benchmark_tflite_fully_connected,
    benchmark_tflite_sub, benchmark_tflite_depthwise_conv2d, benchmark_tflite_max_pool2d,
    benchmark_tflite_per_axis_int8_raw, benchmark_tflite_runner_cache_per_axis_int8_raw,
    run_transform_pipeline, benchmark_matmul, benchmark_transformer_block, benchmark_self_attention,
    benchmark_transformer_encoder, benchmark_transformer_encoder_stack, benchmark_bert_encoder,
    run_bert_safetensors_fixture, evaluate_mnist, benchmark_mnist, @mnist.clear_program_cache,
    run_mnist_cached, benchmark_mnist_cache, probe_program_cache_clear_during_compile,
    probe_concurrent_program, probe_execution_pool, benchmark_mnist_pool, run_named_multi_io,
    benchmark_named_io, run_conv_softmax, benchmark_tiny_cnn, benchmark_tiny_cnn_layout,
    run_grouped_conv, run_preferred_layout_conv,
  )
}
