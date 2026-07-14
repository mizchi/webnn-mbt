///|
fn named_io_operations_per_sample(size : Int) -> Int {
  if size <= 64 {
    20
  } else if size <= 256 {
    10
  } else {
    5
  }
}

///|
fn named_io_values(size : Int, offset : Int) -> Array[Float] {
  Array::makei(size, fn(index) {
    Float::from_int((index + offset) % 17 - 8) / Float::from_int(8)
  })
}

///|
extern "js" fn make_named_io_benchmark_report(
  size : Int,
  warmup : Int,
  iterations : Int,
  operations_per_sample : Int,
  one_compile_ms : Double,
  two_compile_ms : Double,
  one_prepare_ms : Double,
  two_prepare_ms : Double,
  one_output_samples_ms : Array[Double],
  two_output_samples_ms : Array[Double],
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (size, warmup, iterations, operationsPerSample, oneCompileMs, twoCompileMs, onePrepareMs, twoPrepareMs, oneOutputSamplesMs, twoOutputSamplesMs, maxAbsError) => ({
  #|   size,
  #|   warmup,
  #|   iterations,
  #|   operationsPerSample,
  #|   oneCompileMs,
  #|   twoCompileMs,
  #|   onePrepareMs,
  #|   twoPrepareMs,
  #|   oneOutputSamplesMs,
  #|   twoOutputSamplesMs,
  #|   maxAbsError,
  #| })

///|
priv struct NamedIoMeasurement {
  compile_ms : Double
  prepare_ms : Double
  samples_ms : Array[Double]
  results : Array[@webnn.WebNNNamedValues]
}

///|
async fn measure_named_outputs(
  shape : @shape.Shape,
  x_values : Array[Float],
  y_values : Array[Float],
  output_count : Int,
  warmup : Int,
  iterations : Int,
  operations_per_sample : Int,
) -> NamedIoMeasurement {
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  defer builder.destroy()
  let x = builder.input("x", shape)
  let y = builder.input("y", shape)
  let sum = x.tensor().add(y.tensor())
  let outputs = [builder.output("sum", sum)]
  if output_count == 2 {
    outputs.push(builder.output("sum_copy", sum.reshape(shape)))
  }
  let compile_start = performance_now()
  let session = builder.compile_named([x, y], outputs)
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let inputs = [
    @webnn.WebNNNamedValues::new("y", y_values),
    @webnn.WebNNNamedValues::new("x", x_values),
  ]
  let mut results : Array[@webnn.WebNNNamedValues] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      results = execution.run_named(inputs)
    }
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      results = execution.run_named(inputs)
    }
    samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  { compile_ms, prepare_ms, samples_ms, results }
}

///|
pub async fn named_io(size : Int, warmup : Int, iterations : Int) -> @core.Any {
  if size <= 0 || size > 65536 {
    raise @tensor.TensorError::new(
      "named I/O benchmark size must be between 1 and 65536",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "named I/O benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "named I/O benchmark iterations must be positive",
    )
  }
  let shape = @shape.Shape::new([size])
  let x_values = named_io_values(size, 3)
  let y_values = named_io_values(size, 7)
  let expected = Array::makei(size, fn(index) {
    x_values[index] + y_values[index]
  })
  let operations_per_sample = named_io_operations_per_sample(size)

  let one = measure_named_outputs(
    shape, x_values, y_values, 1, warmup, iterations, operations_per_sample,
  )
  let two = measure_named_outputs(
    shape, x_values, y_values, 2, warmup, iterations, operations_per_sample,
  )
  let maximum_absolute_error = max_abs_error(expected, one.results[0].values())
    .max(max_abs_error(expected, two.results[0].values()))
    .max(max_abs_error(expected, two.results[1].values()))
  make_named_io_benchmark_report(
    size,
    warmup,
    iterations,
    operations_per_sample,
    one.compile_ms,
    two.compile_ms,
    one.prepare_ms,
    two.prepare_ms,
    one.samples_ms,
    two.samples_ms,
    maximum_absolute_error,
  )
}
