///|
/// Register all browser-facing playground API domains.
fn main {
  expose_core_api(
    @compat.compatibility_mode, supported_operators, probe_webnn_capabilities, run_cpu_linear,
    run_add, run_concat, run_linear, run_litert_linear, run_transform_pipeline, run_named_multi_io,
    run_conv_softmax, run_grouped_conv, run_preferred_layout_conv,
  )
  expose_tflite_basic_api(
    run_tflite_add, run_tflite_add_relu6, run_tflite_fully_connected, run_tflite_fully_connected_relu6,
    run_tflite_conv2d, run_tflite_reshape, run_tflite_transpose, run_tflite_depthwise_conv2d,
    run_tflite_average_pool2d, run_tflite_max_pool2d, run_tflite_logistic, run_tflite_tanh,
    run_tflite_quantized_add, run_tflite_quantized_add_raw, run_tflite_per_axis_int8_raw,
    run_tflite_cpu_per_axis_int8_raw, run_tflite_sub, run_tflite_relu_n1_to_1, run_tflite_relu6,
  )
  expose_tflite_runtime_api(
    run_tflite_runner_cache_per_axis_int8_raw, run_tflite_runtime_cache_per_axis_int8_raw,
    run_tflite_runtime_cpu_fallback_per_axis_int8_raw, probe_tflite_runtime_policy,
    probe_tflite_runtime_cpu_fallback_cache, probe_tflite_runtime_byte_budget, run_micro_speech_fixture,
    run_person_detection_fixture, probe_mobilenet_v2_fixture, run_mobilenet_v2_fixture,
    benchmark_mobilenet_v2_fixture, probe_audio_preprocessor_fixture, probe_tflite_runner_cache_concurrency,
    probe_tflite_runner_cache_clear_during_compile, run_tflite_runner_named_add,
    run_tflite_runner_two_input_two_output,
  )
  expose_benchmark_api(
    benchmark_litert_linear, benchmark_tflite_fully_connected, benchmark_tflite_sub,
    benchmark_tflite_depthwise_conv2d, benchmark_tflite_max_pool2d, benchmark_tflite_per_axis_int8_raw,
    benchmark_tflite_runner_cache_per_axis_int8_raw, benchmark_matmul, benchmark_transformer_block,
    benchmark_self_attention, benchmark_transformer_encoder, benchmark_transformer_encoder_stack,
    benchmark_bert_encoder, benchmark_named_io, benchmark_tiny_cnn, benchmark_tiny_cnn_layout,
  )
  expose_mnist_and_concurrency_api(
    run_bert_safetensors_fixture, evaluate_mnist, benchmark_mnist, @mnist.clear_program_cache,
    run_mnist_cached, benchmark_mnist_cache, probe_program_cache_clear_during_compile,
    probe_concurrent_program, probe_execution_pool, benchmark_mnist_pool,
  )
}
