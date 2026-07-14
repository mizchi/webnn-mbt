///|
extern "js" fn performance_now() -> Double =
  #| () => performance.now()

///|
extern "js" fn make_report(
  size : Int,
  warmup : Int,
  iterations : Int,
  batch_size : Int,
  context_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  cpu_samples_ms : Array[Double],
  webnn_fresh_samples_ms : Array[Double],
  webnn_samples_ms : Array[Double],
  cpu_checksum : Float,
  webnn_checksum : Float,
  max_abs_error : Float,
) -> @core.Any =
  #| (size, warmup, iterations, batchSize, contextMs, compileMs, prepareMs, cpuSamplesMs, webnnFreshSamplesMs, webnnSamplesMs, cpuChecksum, webnnChecksum, maxAbsError) => ({
  #|   size,
  #|   warmup,
  #|   iterations,
  #|   batchSize,
  #|   device: "npu",
  #|   contextMs,
  #|   compileMs,
  #|   prepareMs,
  #|   cpuSamplesMs,
  #|   webnnFreshSamplesMs,
  #|   webnnSamplesMs,
  #|   cpuChecksum,
  #|   webnnChecksum,
  #|   maxAbsError,
  #| })

///|
fn matrix_values(size : Int, multiplier : Int) -> Array[Float] {
  Array::makei(size * size, fn(index) {
    let value = (index * multiplier + multiplier) % 17 - 8
    Float::from_int(value) / Float::from_int(8)
  })
}

///|
fn operations_per_sample(size : Int) -> Int {
  if size <= 32 {
    20
  } else if size <= 64 {
    10
  } else if size <= 128 {
    3
  } else {
    1
  }
}

///|
fn checksum(values : Array[Float]) -> Float {
  let mut sum = Float::from_int(0)
  for index, value in values {
    let weight = Float::from_int(index % 7 + 1)
    sum = sum + value * weight
  }
  sum
}

///|
fn max_abs_error(lhs : Array[Float], rhs : Array[Float]) -> Float {
  if lhs.length() != rhs.length() {
    return (1.0e30 : Float)
  }
  let mut maximum = Float::from_int(0)
  for index in 0..<lhs.length() {
    let difference = lhs[index] - rhs[index]
    let absolute = if difference < Float::from_int(0) {
      -difference
    } else {
      difference
    }
    if absolute > maximum {
      maximum = absolute
    }
  }
  maximum
}

///|
pub async fn matmul(size : Int, warmup : Int, iterations : Int) -> @core.Any {
  if size <= 0 || size > 1024 {
    raise @tensor.TensorError::new("benchmark size must be between 1 and 1024")
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("benchmark warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("benchmark iterations must be positive")
  }
  let shape = @shape.Shape::new([size, size])
  let input_values = matrix_values(size, 3)
  let weight_values = matrix_values(size, 5)
  let cpu_input = @cpu.CpuTensor::new(shape, input_values)
  let cpu_weight = @cpu.CpuTensor::new(shape, weight_values)
  let batch_size = operations_per_sample(size)
  let mut cpu_result : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<batch_size {
      cpu_result = cpu_input.matmul(cpu_weight).values()
    }
  }
  let cpu_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<batch_size {
      cpu_result = cpu_input.matmul(cpu_weight).values()
    }
    cpu_samples_ms.push(
      (performance_now() - start) / Double::from_int(batch_size),
    )
  }
  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let input_binding = builder.input("input", shape)
  let input = input_binding.tensor()
  let weight = builder.constant(shape, weight_values)
  let output = input.matmul(weight)
  let compile_start = performance_now()
  let session = builder.compile_single(input_binding, "output", output)
  let compile_ms = performance_now() - compile_start
  let mut webnn_result : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<batch_size {
      webnn_result = session.run(input_values)
    }
  }
  let webnn_fresh_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<batch_size {
      webnn_result = session.run(input_values)
    }
    webnn_fresh_samples_ms.push(
      (performance_now() - start) / Double::from_int(batch_size),
    )
  }
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  for _ in 0..<warmup {
    for _ in 0..<batch_size {
      webnn_result = execution.run(input_values)
    }
  }
  let webnn_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<batch_size {
      webnn_result = execution.run(input_values)
    }
    webnn_samples_ms.push(
      (performance_now() - start) / Double::from_int(batch_size),
    )
  }
  make_report(
    size,
    warmup,
    iterations,
    batch_size,
    context_ms,
    compile_ms,
    prepare_ms,
    cpu_samples_ms,
    webnn_fresh_samples_ms,
    webnn_samples_ms,
    checksum(cpu_result),
    checksum(webnn_result),
    max_abs_error(cpu_result, webnn_result),
  )
}
