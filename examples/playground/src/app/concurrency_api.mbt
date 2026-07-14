///|
extern "js" fn run_program_pair(
  run : (Array[Float]) -> @js.Promise[Array[Float]],
  first : Array[Float],
  second : Array[Float],
) -> @js.Promise[Array[Array[Float]]] =
  #| (run, first, second) => Promise.all([run(first), run(second)])

///|
extern "js" fn run_int_program_pair(
  run : (Array[Int]) -> @js.Promise[Array[Int]],
  first : Array[Int],
  second : Array[Int],
) -> @js.Promise[Array[Array[Int]]] =
  #| (run, first, second) => Promise.all([run(first), run(second)])

///|
extern "js" fn make_program_concurrency_report(
  results : Array[Array[Float]],
  rejected_run_recovered : Bool,
  result_after_rejected_run : Array[Float],
  run_after_destroy_rejected : Bool,
) -> @core.Any =
  #| (results, rejectedRunRecovered, resultAfterRejectedRun, runAfterDestroyRejected) => ({
  #|   results,
  #|   rejectedRunRecovered,
  #|   resultAfterRejectedRun,
  #|   runAfterDestroyRejected,
  #| })

///|
extern "js" fn make_tflite_runner_cache_concurrency_report(
  results : Array[Array[Int]],
  cache_size : Int,
) -> @core.Any =
  #| (results, cacheSize) => ({ results, cacheSize })

///|
extern "js" fn make_execution_pool_report(
  results : Array[Array[Float]],
  pool_size : Int,
  maximum_concurrency : Int,
  run_after_destroy_rejected : Bool,
) -> @core.Any =
  #| (results, poolSize, maximumConcurrency, runAfterDestroyRejected) => ({
  #|   results,
  #|   poolSize,
  #|   maximumConcurrency,
  #|   runAfterDestroyRejected,
  #| })

///|
async fn compile_probe_program(pool_size : Int) -> @webnn.WebNNProgram {
  let input_shape = @shape.Shape::new([1, 2])
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let input_binding = builder.input("input", input_shape)
  let weight = builder.constant(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
  let bias = builder.constant(@shape.Shape::new([2]), [1.0, 1.0])
  let output = @model.Linear::{ weight, bias }.forward(input_binding.tensor())
  builder.compile_program_pool_single(
    input_binding, "output", output, pool_size,
  )
}

///|
#external
type CompileGate

///|
extern "js" fn new_compile_gate() -> CompileGate =
  #| () => {
  #|   let started;
  #|   let release;
  #|   return {
  #|     started: new Promise((resolve) => { started = resolve; }),
  #|     release: new Promise((resolve) => { release = resolve; }),
  #|     signalStarted: () => started(),
  #|     open: () => release(),
  #|   };
  #| }

///|
extern "js" fn compile_gate_signal_started(gate : CompileGate) -> Unit =
  #| (gate) => gate.signalStarted()

///|
extern "js" fn compile_gate_started(gate : CompileGate) -> @js.Promise[Unit] =
  #| (gate) => gate.started

///|
extern "js" fn compile_gate_wait(gate : CompileGate) -> @js.Promise[Unit] =
  #| (gate) => gate.release

///|
extern "js" fn compile_gate_open(gate : CompileGate) -> Unit =
  #| (gate) => gate.open()

///|
extern "js" fn make_program_cache_clear_race_report(
  output : Array[Float],
  cache_hit : Bool,
  cache_size_after_clear : Int,
) -> @core.Any =
  #| (output, cacheHit, cacheSizeAfterClear) => ({
  #|   output,
  #|   cacheHit,
  #|   cacheSizeAfterClear,
  #| })

///|
fn probe_program_cache_clear_during_compile() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let cache = @webnn.WebNNProgramCache::new()
    let key = @webnn.WebNNProgramCacheKey::new(
      "cache-clear-race-v1",
      @compat.DevicePreference::Npu,
      [@shape.Shape::new([1, 2])],
    )
    let gate = new_compile_gate()
    let pending = @js.from_async(async fn() {
      cache.run_or_compile(key, [1.0, 2.0], fn() {
        @js.from_async(async fn() {
          compile_gate_signal_started(gate)
          compile_gate_wait(gate).wait()
          compile_probe_program(1)
        })
      })
    })
    compile_gate_started(gate).wait()
    cache.clear() |> ignore
    compile_gate_open(gate)
    let (output, cache_hit) = pending.wait()
    make_program_cache_clear_race_report(output, cache_hit, cache.length())
  })
}

///|
extern "js" fn make_tflite_runner_cache_clear_race_report(
  output : Array[Float],
  cache_size_after_clear : Int,
  resident_bytes_after_clear : Int,
) -> @core.Any =
  #| (output, cacheSizeAfterClear, residentBytesAfterClear) => ({
  #|   output,
  #|   cacheSizeAfterClear,
  #|   residentBytesAfterClear,
  #| })

///|
/// Verifies that clear() invalidates an in-flight TFLite runner compilation,
/// so the completed runner is used once but never retained afterwards.
fn probe_tflite_runner_cache_clear_during_compile() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = tflite_add_fixture()
    let artifact = @webnn.TfliteModelArtifact::from_bytes(bytes)
    let cache = @webnn.TfliteRunnerCache::new()
    defer ignore(cache.clear())
    let gate = new_compile_gate()
    let pending = @js.from_async(async fn() {
      cache.run_artifact_pool_with_compiler(
        artifact,
        @compat.DevicePreference::Npu,
        1,
        [@webnn.TfliteRunnerInput::float32("input", [1.0, 2.0])],
        fn() {
          @js.from_async(async fn() {
            compile_gate_signal_started(gate)
            compile_gate_wait(gate).wait()
            @webnn.TfliteRunner::compile_pool(
              bytes,
              @compat.DevicePreference::Npu,
              1,
            )
          })
        },
      )
    })
    compile_gate_started(gate).wait()
    cache.clear() |> ignore
    compile_gate_open(gate)
    let outputs = pending.wait()
    if outputs.length() != 1 {
      raise @tensor.TensorError::new(
        "expected exactly one TFLite runner output",
      )
    }
    let output = match outputs[0].values() {
      @webnn.TfliteRunnerValues::Float32(values) => values
      @webnn.TfliteRunnerValues::Quantized(_) =>
        raise @tensor.TensorError::new(
          "expected a float32 TFLite runner output",
        )
    }
    make_tflite_runner_cache_clear_race_report(
      output,
      cache.length(),
      cache.resident_bytes(),
    )
  })
}

///|
fn probe_execution_pool() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let program = compile_probe_program(2)
    let pending = run_program_pair(
      fn(values) { @js.from_async(async fn() { program.run(values) }) },
      [1.0, 2.0],
      [3.0, 4.0],
    )
    program.destroy()
    let results = pending.wait()
    let run_after_destroy_rejected = try program.run([1.0, 2.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    make_execution_pool_report(
      results,
      program.pool_size(),
      program.maximum_concurrency(),
      run_after_destroy_rejected,
    )
  })
}

///|
fn probe_concurrent_program() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let program = compile_probe_program(1)
    let rejected_run_recovered = try program.run([1.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    let result_after_rejected_run = program.run([1.0, 2.0])
    let pending = run_program_pair(
      fn(values) { @js.from_async(async fn() { program.run(values) }) },
      [1.0, 2.0],
      [3.0, 4.0],
    )
    program.destroy()
    let results = pending.wait()
    let run_after_destroy_rejected = try program.run([1.0, 2.0]) catch {
      _ => true
    } noraise {
      _ => false
    }
    make_program_concurrency_report(
      results, rejected_run_recovered, result_after_rejected_run, run_after_destroy_rejected,
    )
  })
}
