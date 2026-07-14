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
