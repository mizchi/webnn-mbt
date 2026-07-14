///|
extern "js" fn benchmark_now() -> Double =
  #| () => performance.now()

///|
fn benchmark_operations_per_sample(batch_size : Int) -> Int {
  if batch_size == 1 {
    5
  } else if batch_size <= 4 {
    3
  } else {
    1
  }
}

///|
fn correct_count(predicted : Array[Int], expected : Array[Int]) -> Int {
  let mut correct = 0
  for index in 0..<expected.length() {
    if predicted[index] == expected[index] {
      correct = correct + 1
    }
  }
  correct
}

///|
extern "js" fn make_benchmark_report(
  batch_size : Int,
  operations_per_sample : Int,
  warmup : Int,
  iterations : Int,
  fixture_ms : Double,
  context_ms : Double,
  graph_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  cpu_samples_ms : Array[Double],
  webnn_fresh_samples_ms : Array[Double],
  webnn_samples_ms : Array[Double],
  cpu_correct : Int,
  webnn_correct : Int,
  prediction_matches : Int,
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (batchSize, operationsPerSample, warmup, iterations, fixtureMs, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnFreshSamplesMs, webnnSamplesMs, cpuCorrect, webnnCorrect, predictionMatches, maxAbsError) => ({
  #|   batchSize,
  #|   operationsPerSample,
  #|   warmup,
  #|   iterations,
  #|   fixtureMs,
  #|   contextMs,
  #|   graphMs,
  #|   compileMs,
  #|   prepareMs,
  #|   cpuSamplesMs,
  #|   webnnFreshSamplesMs,
  #|   webnnSamplesMs,
  #|   cpuCorrect,
  #|   webnnCorrect,
  #|   predictionMatches,
  #|   maxAbsError,
  #| })

///|
pub async fn benchmark(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "MNIST benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "MNIST benchmark iterations must be positive",
    )
  }
  let fixture_start = benchmark_now()
  let fixture = load_fixture()
  let fixture_ms = benchmark_now() - fixture_start
  let inputs = fixture.inputs(batch_size)
  let labels = fixture.expected_labels(batch_size)
  let input_shape = @shape.Shape::new([batch_size, input_size])
  let operations_per_sample = benchmark_operations_per_sample(batch_size)
  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu = cpu_model(fixture.params)
  let mut cpu_logits : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      cpu_logits = cpu.forward(cpu_input).values()
    }
  }
  let cpu_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = benchmark_now()
    for _ in 0..<operations_per_sample {
      cpu_logits = cpu.forward(cpu_input).values()
    }
    cpu_samples_ms.push(
      (benchmark_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  let context_start = benchmark_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = benchmark_now() - context_start
  defer builder.destroy()
  let graph_start = benchmark_now()
  let input_binding = builder.input("input", input_shape)
  let output = webnn_model(builder, fixture.params).forward(
    input_binding.tensor(),
  )
  let graph_ms = benchmark_now() - graph_start
  let compile_start = benchmark_now()
  let session = builder.compile_single(input_binding, "output", output)
  let compile_ms = benchmark_now() - compile_start
  let mut webnn_logits : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      webnn_logits = session.run(inputs)
    }
  }
  let webnn_fresh_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = benchmark_now()
    for _ in 0..<operations_per_sample {
      webnn_logits = session.run(inputs)
    }
    webnn_fresh_samples_ms.push(
      (benchmark_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  let prepare_start = benchmark_now()
  let execution = session.prepare()
  let prepare_ms = benchmark_now() - prepare_start
  defer execution.destroy()
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      webnn_logits = execution.run(inputs)
    }
  }
  let webnn_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = benchmark_now()
    for _ in 0..<operations_per_sample {
      webnn_logits = execution.run(inputs)
    }
    webnn_samples_ms.push(
      (benchmark_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  let cpu_predictions = predictions(cpu_logits, batch_size)
  let webnn_predictions = predictions(webnn_logits, batch_size)
  make_benchmark_report(
    batch_size,
    operations_per_sample,
    warmup,
    iterations,
    fixture_ms,
    context_ms,
    graph_ms,
    compile_ms,
    prepare_ms,
    cpu_samples_ms,
    webnn_fresh_samples_ms,
    webnn_samples_ms,
    correct_count(cpu_predictions, labels),
    correct_count(webnn_predictions, labels),
    correct_count(cpu_predictions, webnn_predictions),
    max_abs_error(cpu_logits, webnn_logits),
  )
}
