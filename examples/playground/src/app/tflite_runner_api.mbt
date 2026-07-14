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
