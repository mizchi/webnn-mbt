///| The graph identity covers the MLP topology/layout and the exact fixture

///|
/// weights. Changing either must produce a new cache namespace.
let mnist_graph_hash : String = "mnist-mlp-784x128x10-v1+weights-sha256:1f8c4f957e8f71e545f06d64ddf1a132202229e24c374e9ce6bc426f5a396424"

///|
let cached_fixture : Ref[Fixture?] = { val: None }

///|
let program_cache : @webnn.WebNNProgramCache = @webnn.WebNNProgramCache::new()

///|
async fn load_cached_fixture() -> Fixture {
  match cached_fixture.val {
    Some(fixture) => fixture
    None => {
      let fixture = load_fixture()
      cached_fixture.val = Some(fixture)
      fixture
    }
  }
}

///|
fn program_key(
  input_shape : @shape.Shape,
  preference : @compat.DevicePreference,
) -> @webnn.WebNNProgramCacheKey raise @tensor.TensorError {
  @webnn.WebNNProgramCacheKey::new(mnist_graph_hash, preference, [input_shape])
}

///|
async fn compile_program(
  params : @model.MlpParams,
  input_shape : @shape.Shape,
  preference : @compat.DevicePreference,
) -> @webnn.WebNNProgram {
  compile_program_pool(params, input_shape, preference, 1)
}

///|
async fn compile_program_pool(
  params : @model.MlpParams,
  input_shape : @shape.Shape,
  preference : @compat.DevicePreference,
  pool_size : Int,
) -> @webnn.WebNNProgram {
  let builder = @webnn.WebNNGraphBuilder::new(preference)
  let input_binding = builder.input("input", input_shape)
  let output = webnn_model(builder, params).forward(input_binding.tensor()) catch {
    error => {
      builder.destroy()
      raise error
    }
  }
  builder.compile_program_pool_single(
    input_binding, "output", output, pool_size,
  )
}

///|
async fn cached_run(
  params : @model.MlpParams,
  input_shape : @shape.Shape,
  preference : @compat.DevicePreference,
  inputs : Array[Float],
) -> (Array[Float], Bool) {
  let key = program_key(input_shape, preference)
  program_cache.run_or_compile(key, inputs, fn() {
    @js.from_async(async fn() {
      compile_program(params, input_shape, preference)
    })
  })
}

///|
extern "js" fn make_cached_report(
  batch_size : Int,
  cache_hit : Bool,
  cache_size : Int,
  logits : Array[Float],
  predicted : Array[Int],
) -> @core.Any =
  #| (batchSize, cacheHit, cacheSize, logits, predictions) => ({
  #|   batchSize,
  #|   cacheHit,
  #|   cacheSize,
  #|   logits,
  #|   predictions,
  #| })

///|
pub async fn run_cached(batch_size : Int) -> @core.Any {
  let fixture = load_cached_fixture()
  let inputs = fixture.inputs(batch_size)
  let input_shape = @shape.Shape::new([batch_size, input_size])
  let (logits, cache_hit) = cached_run(
    fixture.params,
    input_shape,
    @compat.DevicePreference::Npu,
    inputs,
  )
  make_cached_report(
    batch_size,
    cache_hit,
    program_cache.length(),
    logits,
    predictions(logits, batch_size),
  )
}

///|
pub fn clear_program_cache() -> Int {
  program_cache.clear()
}

///|
extern "js" fn make_cache_benchmark_report(
  batch_size : Int,
  operations_per_sample : Int,
  warmup : Int,
  iterations : Int,
  cold_samples_ms : Array[Double],
  hit_samples_ms : Array[Double],
  cold_misses : Int,
  cache_size : Int,
  prediction_matches : Int,
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (batchSize, operationsPerSample, warmup, iterations, coldSamplesMs, hitSamplesMs, coldMisses, cacheSize, predictionMatches, maxAbsError) => ({
  #|   batchSize,
  #|   operationsPerSample,
  #|   warmup,
  #|   iterations,
  #|   coldSamplesMs,
  #|   hitSamplesMs,
  #|   coldMisses,
  #|   cacheSize,
  #|   predictionMatches,
  #|   maxAbsError,
  #| })

///| Measures the application-level cache path. Fixture loading and CPU

///|
/// reference calculation are intentionally outside both timed regions.
pub async fn benchmark_cache(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if warmup < 0 {
    raise @tensor.TensorError::new(
      "MNIST cache benchmark warmup must be non-negative",
    )
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new(
      "MNIST cache benchmark iterations must be positive",
    )
  }
  let fixture = load_cached_fixture()
  let inputs = fixture.inputs(batch_size)
  let input_shape = @shape.Shape::new([batch_size, input_size])
  let preference = @compat.DevicePreference::Npu
  let operations_per_sample = benchmark_operations_per_sample(batch_size)
  let cpu_logits = cpu_model(fixture.params)
    .forward(@cpu.CpuTensor::new(input_shape, inputs))
    .values()
  let cold_samples_ms : Array[Double] = []
  let mut cold_misses = 0
  let mut webnn_logits : Array[Float] = []
  for _ in 0..<iterations {
    program_cache.clear() |> ignore
    let start = benchmark_now()
    let (logits, cache_hit) = cached_run(
      fixture.params,
      input_shape,
      preference,
      inputs,
    )
    webnn_logits = logits
    cold_samples_ms.push(benchmark_now() - start)
    if !cache_hit {
      cold_misses = cold_misses + 1
    }
  }
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      let (logits, _) = cached_run(
        fixture.params,
        input_shape,
        preference,
        inputs,
      )
      webnn_logits = logits
    }
  }
  let hit_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = benchmark_now()
    for _ in 0..<operations_per_sample {
      let (logits, _) = cached_run(
        fixture.params,
        input_shape,
        preference,
        inputs,
      )
      webnn_logits = logits
    }
    hit_samples_ms.push(
      (benchmark_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_cache_benchmark_report(
    batch_size,
    operations_per_sample,
    warmup,
    iterations,
    cold_samples_ms,
    hit_samples_ms,
    cold_misses,
    program_cache.length(),
    correct_count(
      predictions(cpu_logits, batch_size),
      predictions(webnn_logits, batch_size),
    ),
    max_abs_error(cpu_logits, webnn_logits),
  )
}
