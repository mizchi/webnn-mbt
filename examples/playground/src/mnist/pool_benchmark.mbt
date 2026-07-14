///|
extern "js" fn run_concurrent_requests(
  run : () -> @js.Promise[Array[Float]],
  request_count : Int,
) -> @js.Promise[Array[Array[Float]]] =
  #| (run, requestCount) => Promise.all(
  #|   Array.from({ length: requestCount }, () => run()),
  #| )

///|
extern "js" fn make_pool_benchmark_report(
  pool_size : Int,
  request_count : Int,
  warmup : Int,
  iterations : Int,
  setup_ms : Double,
  samples_ms : Array[Double],
  maximum_concurrency : Int,
  prediction_matches : Int,
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (poolSize, requestCount, warmup, iterations, setupMs, samplesMs, maximumConcurrency, predictionMatches, maxAbsError) => ({
  #|   poolSize,
  #|   requestCount,
  #|   warmup,
  #|   iterations,
  #|   setupMs,
  #|   samplesMs,
  #|   maximumConcurrency,
  #|   predictionMatches,
  #|   maxAbsError,
  #| })

///|
pub async fn benchmark_pool(
  pool_size : Int,
  request_count : Int,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if pool_size <= 0 {
    raise @tensor.TensorError::new("MNIST execution pool size must be positive")
  }
  if request_count <= 0 {
    raise @tensor.TensorError::new(
      "MNIST execution pool request count must be positive",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "MNIST execution pool warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "MNIST execution pool iterations must be positive",
    )
  }
  let fixture = load_cached_fixture()
  let inputs = fixture.inputs(1)
  let input_shape = @shape.Shape::new([1, input_size])
  let cpu_logits = cpu_model(fixture.params)
    .forward(@cpu.CpuTensor::new(input_shape, inputs))
    .values()
  let setup_start = benchmark_now()
  let program = compile_program_pool(
    fixture.params,
    input_shape,
    @compat.DevicePreference::Npu,
    pool_size,
  )
  let setup_ms = benchmark_now() - setup_start
  defer program.destroy()
  let run_group = fn() {
    run_concurrent_requests(
      fn() { @js.from_async(async fn() { program.run(inputs) }) },
      request_count,
    )
  }
  let mut latest_results : Array[Array[Float]] = []
  for _ in 0..<warmup {
    latest_results = run_group().wait()
  }
  let samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = benchmark_now()
    latest_results = run_group().wait()
    samples_ms.push(benchmark_now() - start)
  }
  let cpu_prediction = predictions(cpu_logits, 1)[0]
  let mut prediction_matches = 0
  let mut maximum_absolute_error = Float::from_int(0)
  for logits in latest_results {
    if predictions(logits, 1)[0] == cpu_prediction {
      prediction_matches = prediction_matches + 1
    }
    let error = max_abs_error(cpu_logits, logits)
    if error > maximum_absolute_error {
      maximum_absolute_error = error
    }
  }
  make_pool_benchmark_report(
    pool_size,
    request_count,
    warmup,
    iterations,
    setup_ms,
    samples_ms,
    program.maximum_concurrency(),
    prediction_matches,
    maximum_absolute_error,
  )
}
