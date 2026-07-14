///|
/// Register core graph and capability APIs on the browser test surface.
extern "js" fn expose_core_api(
  compatibility_mode : () -> String,
  supported_operators : () -> @js.Promise[Array[String]],
  probe_webnn_capabilities : () -> @js.Promise[@core.Any],
  run_cpu_linear : () -> @js.Promise[Array[Float]],
  run_add : () -> @js.Promise[Array[Float]],
  run_concat : () -> @js.Promise[Array[Float]],
  run_linear : () -> @js.Promise[Array[Float]],
  run_litert_linear : () -> @js.Promise[Array[Float]],
  run_transform_pipeline : () -> @js.Promise[Array[Float]],
  run_named_multi_io : () -> @js.Promise[@core.Any],
  run_conv_softmax : () -> @js.Promise[Array[Float]],
  run_grouped_conv : () -> @js.Promise[Array[Float]],
  run_preferred_layout_conv : () -> @js.Promise[@core.Any],
) -> Unit =
  #| (compatibilityMode, supportedOperators, probeWebnnCapabilities, runCpuLinear, runAdd, runConcat, runLinear, runLiteRtLinear, runTransformPipeline, runNamedMultiIo, runConvSoftmax, runGroupedConv, runPreferredLayoutConv) => {
  #|   globalThis.webnnPlayground ??= {};
  #|   Object.assign(globalThis.webnnPlayground, {
  #|     compatibilityMode, supportedOperators, probeWebnnCapabilities,
  #|     runCpuLinear, runAdd, runConcat, runLinear, runLiteRtLinear,
  #|     runTransformPipeline, runNamedMultiIo, runConvSoftmax,
  #|     runGroupedConv, runPreferredLayoutConv,
  #|   });
  #| }

///|
/// Register direct TFLite and quantized execution APIs.
extern "js" fn expose_tflite_basic_api(
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
  run_tflite_cpu_per_axis_int8_raw : () -> @js.Promise[Array[Int]],
  run_tflite_sub : () -> @js.Promise[Array[Float]],
  run_tflite_relu_n1_to_1 : () -> @js.Promise[Array[Float]],
  run_tflite_relu6 : () -> @js.Promise[Array[Float]],
) -> Unit =
  #| (runTfliteAdd, runTfliteAddRelu6, runTfliteFullyConnected, runTfliteFullyConnectedRelu6, runTfliteConv2d, runTfliteReshape, runTfliteTranspose, runTfliteDepthwiseConv2d, runTfliteAveragePool2d, runTfliteMaxPool2d, runTfliteLogistic, runTfliteTanh, runTfliteQuantizedAdd, runTfliteQuantizedAddRaw, runTflitePerAxisInt8Raw, runTfliteCpuPerAxisInt8Raw, runTfliteSub, runTfliteReluN1To1, runTfliteRelu6) => {
  #|   globalThis.webnnPlayground ??= {};
  #|   Object.assign(globalThis.webnnPlayground, {
  #|     runTfliteAdd, runTfliteAddRelu6, runTfliteFullyConnected,
  #|     runTfliteFullyConnectedRelu6, runTfliteConv2d, runTfliteReshape,
  #|     runTfliteTranspose, runTfliteDepthwiseConv2d, runTfliteAveragePool2d,
  #|     runTfliteMaxPool2d, runTfliteLogistic, runTfliteTanh,
  #|     runTfliteQuantizedAdd, runTfliteQuantizedAddRaw,
  #|     runTflitePerAxisInt8Raw, runTfliteCpuPerAxisInt8Raw,
  #|     runTfliteSub, runTfliteReluN1To1, runTfliteRelu6,
  #|   });
  #| }

///|
/// Register cached TFLite runtime and real-model fixture APIs.
extern "js" fn expose_tflite_runtime_api(
  run_tflite_runner_cache_per_axis_int8_raw : () -> @js.Promise[@core.Any],
  run_tflite_runtime_cache_per_axis_int8_raw : () -> @js.Promise[@core.Any],
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
) -> Unit =
  #| (runTfliteRunnerCachePerAxisInt8Raw, runTfliteRuntimeCachePerAxisInt8Raw, runTfliteRuntimeCpuFallbackPerAxisInt8Raw, probeTfliteRuntimePolicy, probeTfliteRuntimeCpuFallbackCache, probeTfliteRuntimeByteBudget, runMicroSpeechFixture, runPersonDetectionFixture, probeMobileNetV2Fixture, runMobileNetV2Fixture, benchmarkMobileNetV2Fixture, probeAudioPreprocessorFixture, probeTfliteRunnerCacheConcurrency, probeTfliteRunnerCacheClearDuringCompile, runTfliteRunnerNamedAdd, runTfliteRunnerTwoInputTwoOutput) => {
  #|   globalThis.webnnPlayground ??= {};
  #|   Object.assign(globalThis.webnnPlayground, {
  #|     runTfliteRunnerCachePerAxisInt8Raw, runTfliteRuntimeCachePerAxisInt8Raw,
  #|     runTfliteRuntimeCpuFallbackPerAxisInt8Raw, probeTfliteRuntimePolicy,
  #|     probeTfliteRuntimeCpuFallbackCache, probeTfliteRuntimeByteBudget,
  #|     runMicroSpeechFixture, runPersonDetectionFixture,
  #|     probeMobileNetV2Fixture, runMobileNetV2Fixture,
  #|     benchmarkMobileNetV2Fixture, probeAudioPreprocessorFixture,
  #|     probeTfliteRunnerCacheConcurrency,
  #|     probeTfliteRunnerCacheClearDuringCompile, runTfliteRunnerNamedAdd,
  #|     runTfliteRunnerTwoInputTwoOutput,
  #|   });
  #| }

///|
/// Register performance benchmark APIs.
extern "js" fn expose_benchmark_api(
  benchmark_litert_linear : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_fully_connected : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_sub : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_depthwise_conv2d : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_max_pool2d : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_per_axis_int8_raw : (Int, Int) -> @js.Promise[@core.Any],
  benchmark_tflite_runner_cache_per_axis_int8_raw : (Int, Int) -> @js.Promise[
    @core.Any,
  ],
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
  benchmark_named_io : (Int, Int, Int) -> @js.Promise[@core.Any],
  benchmark_tiny_cnn : (Int, Int, Int) -> @js.Promise[@core.Any],
  benchmark_tiny_cnn_layout : (String, Int, Int, Int) -> @js.Promise[@core.Any],
) -> Unit =
  #| (benchmarkLiteRtLinear, benchmarkTfliteFullyConnected, benchmarkTfliteSub, benchmarkTfliteDepthwiseConv2d, benchmarkTfliteMaxPool2d, benchmarkTflitePerAxisInt8Raw, benchmarkTfliteRunnerCachePerAxisInt8Raw, benchmarkMatmul, benchmarkTransformerBlock, benchmarkSelfAttention, benchmarkTransformerEncoder, benchmarkTransformerEncoderStack, benchmarkBertEncoder, benchmarkNamedIo, benchmarkTinyCnn, benchmarkTinyCnnLayout) => {
  #|   globalThis.webnnPlayground ??= {};
  #|   Object.assign(globalThis.webnnPlayground, {
  #|     benchmarkLiteRtLinear, benchmarkTfliteFullyConnected,
  #|     benchmarkTfliteSub, benchmarkTfliteDepthwiseConv2d,
  #|     benchmarkTfliteMaxPool2d, benchmarkTflitePerAxisInt8Raw,
  #|     benchmarkTfliteRunnerCachePerAxisInt8Raw, benchmarkMatmul,
  #|     benchmarkTransformerBlock, benchmarkSelfAttention,
  #|     benchmarkTransformerEncoder, benchmarkTransformerEncoderStack,
  #|     benchmarkBertEncoder, benchmarkNamedIo, benchmarkTinyCnn,
  #|     benchmarkTinyCnnLayout,
  #|   });
  #| }

///|
/// Register MNIST, BERT fixture, and concurrency probe APIs.
extern "js" fn expose_mnist_and_concurrency_api(
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
) -> Unit =
  #| (runBertSafetensorsFixture, evaluateMnist, benchmarkMnist, clearMnistCache, runMnistCached, benchmarkMnistCache, probeProgramCacheClearDuringCompile, probeConcurrentProgram, probeExecutionPool, benchmarkMnistPool) => {
  #|   globalThis.webnnPlayground ??= {};
  #|   Object.assign(globalThis.webnnPlayground, {
  #|     runBertSafetensorsFixture, evaluateMnist, benchmarkMnist,
  #|     clearMnistCache, runMnistCached, benchmarkMnistCache,
  #|     probeProgramCacheClearDuringCompile, probeConcurrentProgram,
  #|     probeExecutionPool, benchmarkMnistPool,
  #|   });
  #| }
