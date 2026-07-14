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
