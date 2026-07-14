///|
fn litert_linear_model() -> @litert.LiteRtModel raise {
  let input_shape = @shape.Shape::new([1, 2])
  let weight_shape = @shape.Shape::new([2, 2])
  let output_shape = @shape.Shape::new([1, 2])
  @litert.LiteRtModel::new(
    [
      @litert.LiteRtValue::input("input", input_shape),
      @litert.LiteRtValue::constant("weight", weight_shape, [1.0, 2.0, 3.0, 4.0]),
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
}

///|
extern "js" fn make_litert_benchmark_report(
  warmup : Int,
  iterations : Int,
  operations_per_sample : Int,
  context_ms : Double,
  lower_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  samples_ms : Array[Double],
  output : Array[Float],
) -> @core.Any =
  #| (warmup, iterations, operationsPerSample, contextMs, lowerMs, compileMs, prepareMs, samplesMs, output) => ({
  #|   warmup,
  #|   iterations,
  #|   operationsPerSample,
  #|   contextMs,
  #|   lowerMs,
  #|   compileMs,
  #|   prepareMs,
  #|   samplesMs,
  #|   output,
  #| })

///|
pub async fn litert_linear(warmup : Int, iterations : Int) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "LiteRT benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "LiteRT benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(litert_linear_model())
  let lower_ms = performance_now() - lower_start
  let inputs = lowered.inputs()
  let outputs = lowered.outputs()
  let compile_start = performance_now()
  let session = builder.compile_single(
    inputs[0],
    outputs[0].name(),
    outputs[0].tensor(),
  )
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let input_values : Array[Float] = [1.0, 2.0]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_litert_benchmark_report(
    warmup, iterations, operations_per_sample, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
extern "js" fn tflite_fully_connected_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAAHABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAAJAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAOwAAADwAAAA9AAAAAQAAAAcAAAATAAAAHwAAACoAAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAEAAAAAIAAAABAAAAAgAAAAUAAABpbnB1dAAMABAABAAAAAgADAAAAA4AAAAMAAAAAQAAABAAAAACAAAAAgAAAAIAAAAHAAAAd2VpZ2h0cwAMABAABAAAAAgADAAMAAAADAAAAAIAAAAMAAAAAQAAAAIAAAAEAAAAYmlhcwAADAAQAAQAAAAIAAwAAAAOAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAAAQAAAACgAMAAAABAAIAAAADAAAAAgAAAAUAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAAwAAABAAAAAYAAAAOAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAABAAAAAAAIA/AABAQAAAAEAAAIBABgAIAAQAAAAIAAAABAAAAAgAAAAAAIA/AACAPw=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_sub_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAcAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAKQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACIAAAAkAAAAJQAAAADAAAAGAAAADgAAABYAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAAAgAAAAAAAAABAAAAAQAAAAIAAAABAAAAEAAAAAoAEAAAAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAMAAAABgAIAAAAAAAIAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_depthwise_conv2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAC4AQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAABAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAADkAAAA6AAAAOwAAAAEAAAAHAAAAEwAAAB8AAAAoAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAADAAAAAwAAAAIAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAABAAAAAAAAAAQAAAABAAAAAgAAAAIAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAgAAAAAAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAACAAAAAgAAAAIAAAABAAAAAAAAAAEAAAADAAAAAQAAABQAAAAOABgABAAIAAwAAAAUAAAAEAAAAAAAAAAQAAAAHAAAAAAAAAAwAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAEgAgAAQACAAMABAAFAAYABwAAAAUAAAAAQAAAAEAAAABAAAAAQAAAAAAAAABAAAAAQAAAAMAAAAUAAAAIAAAAFAAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAACAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEAAAOBAAAAAQQYACAAEAAAACAAAAAQAAAAIAAAAAACAPwAAIME="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_max_pool2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAABEAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAEQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAAAAhAAAAIgAAAACAAAAFAAAAEQAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAQAAAABAAAAAwAAAAMAAAABAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAAEAAAAAQAAAAIAAAACAAAAAQAAAAEAAAAAAAAAAQAAAAEAAAABAAAAFAAAAA4AGAAEAAgADAAAABQAAAAQAAAAAAAAABAAAAAUAAAAAAAAACQAAAABAAAAAAAAAAEAAAABAAAAEAAcAAQACAAMABAAFAAYABAAAAABAAAAAQAAAAEAAAACAAAAAgAAAAAAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_int8_per_axis_runtime_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAACQAAAAAAAAAAAAAAzAEAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAAAAAAAAAAAkAQAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAAACBAAAEAAQAAAAAAAEAAgAAAAMABAAAAAMAAAAGAAAAAEAAAACAAAAAAAAPwAAgD4AAAAAAgAAAAEAAAAAAAAA/v////////8QABAAAAAAAAQACAAAAAwAEAAAAAwAAAAUAAAAAQAAAAIAAAAAAAA/AACAPgIAAAABAAAAAAAAAP7/////////"),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn make_tflite_benchmark_report(
  warmup : Int,
  iterations : Int,
  operations_per_sample : Int,
  parse_ms : Double,
  context_ms : Double,
  lower_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  samples_ms : Array[Double],
  output : Array[Float],
) -> @core.Any =
  #| (warmup, iterations, operationsPerSample, parseMs, contextMs, lowerMs, compileMs, prepareMs, samplesMs, output) => ({
  #|   warmup,
  #|   iterations,
  #|   operationsPerSample,
  #|   parseMs,
  #|   contextMs,
  #|   lowerMs,
  #|   compileMs,
  #|   prepareMs,
  #|   samplesMs,
  #|   output,
  #| })

///|
pub async fn tflite_fully_connected(
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let parse_start = performance_now()
  let model = @litert.TfliteModel::parse(tflite_fully_connected_fixture())
  let parse_ms = performance_now() - parse_start
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(model)
  let lower_ms = performance_now() - lower_start
  let inputs = lowered.inputs()
  let outputs = lowered.outputs()
  let compile_start = performance_now()
  let session = builder.compile_single(
    inputs[0],
    outputs[0].name(),
    outputs[0].tensor(),
  )
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let input_values : Array[Float] = [1.0, 2.0]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tflite_benchmark_report(
    warmup, iterations, operations_per_sample, parse_ms, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
pub async fn tflite_sub(warmup : Int, iterations : Int) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let parse_start = performance_now()
  let model = @litert.TfliteModel::parse(tflite_sub_fixture())
  let parse_ms = performance_now() - parse_start
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(model)
  let lower_ms = performance_now() - lower_start
  let output_bindings : Array[@webnn.WebNNOutput] = []
  for output in lowered.outputs() {
    output_bindings.push(builder.output(output.name(), output.tensor()))
  }
  let compile_start = performance_now()
  let session = builder.compile_named(lowered.inputs(), output_bindings)
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let input_values = [
    @webnn.WebNNNamedValues::new("tensor_1", [1.0, 10.0]),
    @webnn.WebNNNamedValues::new("tensor_0", [3.0, 20.0]),
  ]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      output = execution.run_named(input_values)[0].values()
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      output = execution.run_named(input_values)[0].values()
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tflite_benchmark_report(
    warmup, iterations, operations_per_sample, parse_ms, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
pub async fn tflite_depthwise_conv2d(
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let parse_start = performance_now()
  let model = @litert.TfliteModel::parse(tflite_depthwise_conv2d_fixture())
  let parse_ms = performance_now() - parse_start
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(model)
  let lower_ms = performance_now() - lower_start
  let inputs = lowered.inputs()
  let outputs = lowered.outputs()
  let compile_start = performance_now()
  let session = builder.compile_single(
    inputs[0],
    outputs[0].name(),
    outputs[0].tensor(),
  )
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let input_values : Array[Float] = [
    1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0,
    8.0, 80.0, 9.0, 90.0,
  ]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tflite_benchmark_report(
    warmup, iterations, operations_per_sample, parse_ms, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
pub async fn tflite_max_pool2d(warmup : Int, iterations : Int) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let parse_start = performance_now()
  let model = @litert.TfliteModel::parse(tflite_max_pool2d_fixture())
  let parse_ms = performance_now() - parse_start
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(model)
  let lower_ms = performance_now() - lower_start
  let inputs = lowered.inputs()
  let outputs = lowered.outputs()
  let compile_start = performance_now()
  let session = builder.compile_single(
    inputs[0],
    outputs[0].name(),
    outputs[0].tensor(),
  )
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let input_values : Array[Float] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      output = execution.run(input_values)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tflite_benchmark_report(
    warmup, iterations, operations_per_sample, parse_ms, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
pub async fn tflite_per_axis_int8_raw(
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite benchmark iterations must be positive",
    )
  }
  let operations_per_sample = 20
  let parse_start = performance_now()
  let runtime = @litert.TfliteModel::parse_runtime(
    tflite_int8_per_axis_runtime_add_fixture(),
  )
  let parse_ms = performance_now() - parse_start
  let runtime_inputs = runtime.inputs()
  let runtime_outputs = runtime.outputs()
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let lower_start = performance_now()
  let lowered = builder.lower_litert(runtime.model())
  let lower_ms = performance_now() - lower_start
  let inputs = lowered.inputs()
  let outputs = lowered.outputs()
  let compile_start = performance_now()
  let session = builder.compile_single(
    inputs[0],
    outputs[0].name(),
    outputs[0].tensor(),
  )
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let raw_input : Array[Int] = [3, 2]
  let mut output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      let input = runtime_inputs[0].dequantize(raw_input)
      output = execution.run(input)
      let _ = runtime_outputs[0].requantize(output)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      let input = runtime_inputs[0].dequantize(raw_input)
      output = execution.run(input)
      let _ = runtime_outputs[0].requantize(output)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tflite_benchmark_report(
    warmup, iterations, operations_per_sample, parse_ms, context_ms, lower_ms, compile_ms,
    prepare_ms, samples_ms, output,
  )
}

///|
extern "js" fn make_tflite_runner_cache_benchmark_report(
  warmup : Int,
  iterations : Int,
  hit_operations_per_sample : Int,
  cold_samples_ms : Array[Double],
  hit_samples_ms : Array[Double],
  output : Array[Int],
  cache_size : Int,
) -> @core.Any =
  #| (warmup, iterations, hitOperationsPerSample, coldSamplesMs, hitSamplesMs, output, cacheSize) => ({
  #|   warmup,
  #|   iterations,
  #|   hitOperationsPerSample,
  #|   coldSamplesMs,
  #|   hitSamplesMs,
  #|   output,
  #|   cacheSize,
  #| })

///|
fn tflite_runner_raw_output(
  outputs : Array[@webnn.TfliteRunnerOutput],
) -> Array[Int] raise @tensor.TensorError {
  if outputs.length() != 1 {
    raise @tensor.TensorError::new(
      "TFLite runner benchmark expects exactly one output",
    )
  }
  match outputs[0].values() {
    @webnn.TfliteRunnerValues::Quantized(values) => values
    @webnn.TfliteRunnerValues::Float32(_) =>
      raise @tensor.TensorError::new(
        "TFLite runner benchmark expects a quantized output",
      )
  }
}

///|
fn tflite_runner_per_axis_input() -> Array[@webnn.TfliteRunnerInput] raise {
  [@webnn.TfliteRunnerInput::quantized("tensor_0", [3, 2])]
}

///|
/// Compare TFLiteRunner cold compilation with SHA-256 keyed cache hits.
pub async fn tflite_runner_cache_per_axis_int8_raw(
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "TFLite runner cache benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "TFLite runner cache benchmark iterations must be positive",
    )
  }
  let bytes = tflite_int8_per_axis_runtime_add_fixture()
  let cache = @webnn.TfliteRunnerCache::new()
  defer ignore(cache.clear())
  let mut output : Array[Int] = []
  for _ in 0..<warmup {
    output = tflite_runner_raw_output(
      cache.run(
        bytes,
        @compat.DevicePreference::Npu,
        tflite_runner_per_axis_input(),
      ),
    )
  }
  let cold_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let _ = cache.clear()
    let start = performance_now()
    output = tflite_runner_raw_output(
      cache.run(
        bytes,
        @compat.DevicePreference::Npu,
        tflite_runner_per_axis_input(),
      ),
    )
    cold_samples_ms.push(performance_now() - start)
  }
  let _ = cache.clear()
  output = tflite_runner_raw_output(
    cache.run(
      bytes,
      @compat.DevicePreference::Npu,
      tflite_runner_per_axis_input(),
    ),
  )
  let hit_operations_per_sample = 20
  for _ in 0..<warmup {
    for _ in 0..<hit_operations_per_sample {
      output = tflite_runner_raw_output(
        cache.run(
          bytes,
          @compat.DevicePreference::Npu,
          tflite_runner_per_axis_input(),
        ),
      )
    }
  }
  let hit_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<hit_operations_per_sample {
      output = tflite_runner_raw_output(
        cache.run(
          bytes,
          @compat.DevicePreference::Npu,
          tflite_runner_per_axis_input(),
        ),
      )
    }
    hit_samples_ms.push(
      (performance_now() - start) / Double::from_int(hit_operations_per_sample),
    )
  }
  make_tflite_runner_cache_benchmark_report(
    warmup,
    iterations,
    hit_operations_per_sample,
    cold_samples_ms,
    hit_samples_ms,
    output,
    cache.length(),
  )
}
