///|
/// Runtime values accepted by and returned from TfliteRunner.
///
/// The WebNN graph always uses float32 internally. Quantized keeps the
/// caller-facing UINT8/INT8 representation at the TFLite boundary.
pub enum TfliteRunnerValues {
  Float32(Array[Float])
  Quantized(Array[Int])
}

///|
pub struct TfliteRunnerInput {
  name_ : String
  values_ : TfliteRunnerValues
}

///|
pub struct TfliteRunnerOutput {
  name_ : String
  values_ : TfliteRunnerValues
}

///|
pub struct TfliteRunner {
  program : WebNNProgram
  inputs : Array[@litert.TfliteRuntimeTensor]
  outputs : Array[@litert.TfliteRuntimeTensor]
}

///|
pub struct TfliteCpuRunner {
  runtime : @litert.TfliteRuntimeModel
}

///|
/// Immutable TFLite bytes with a SHA-256 digest computed once.
///
/// Construct this artifact once when a model is loaded, then pass it to
/// WebNNRuntime::run_prepared_tflite to avoid re-hashing large FlatBuffers on
/// every cache lookup.
pub struct TfliteModelArtifact {
  bytes_ : Bytes
  digest_ : String
}

///|
pub struct TfliteRunnerCacheKey {
  model_digest_ : String
  device_ : @compat.DevicePreference
  execution_pool_size_ : Int
}

///|
pub struct TfliteRunnerCacheMetrics {
  hits_ : Int
  misses_ : Int
  evictions_ : Int
  entries_ : Int
  resident_bytes_ : Int
  byte_capacity_ : Int
}

///|
pub struct TfliteRunnerCache {
  entries : Map[String, TfliteRunner]
  entry_bytes : Map[String, Int]
  last_access : Map[String, Int]
  capacity_ : Int
  byte_capacity_ : Int
  generation : Ref[Int]
  mut resident_bytes_ : Int
  mut access_counter : Int
  mut hits_ : Int
  mut misses_ : Int
  mut evictions_ : Int
}

///|
pub struct TfliteCpuRunnerCacheMetrics {
  hits_ : Int
  misses_ : Int
  evictions_ : Int
  entries_ : Int
  resident_bytes_ : Int
  byte_capacity_ : Int
}

///|
pub struct TfliteCpuRunnerCache {
  entries : Map[String, TfliteCpuRunner]
  entry_bytes : Map[String, Int]
  last_access : Map[String, Int]
  capacity_ : Int
  byte_capacity_ : Int
  mut resident_bytes_ : Int
  mut access_counter : Int
  mut hits_ : Int
  mut misses_ : Int
  mut evictions_ : Int
}

///|
priv enum CachedTfliteRunner {
  Retained(TfliteRunner)
  Transient(TfliteRunner)
}

///|
let default_tflite_runner_cache_capacity : Int = 32

///|
let default_tflite_runner_cache_byte_capacity : Int = 64 * 1024 * 1024

///|
fn ensure_tflite_runner_name(name : String) -> Unit raise @tensor.TensorError {
  if name == "" {
    raise @tensor.TensorError::new("TFLite runner value name must not be empty")
  }
}

///|
pub fn TfliteRunnerInput::float32(
  name : String,
  values : Array[Float],
) -> TfliteRunnerInput raise @tensor.TensorError {
  ensure_tflite_runner_name(name)
  { name_: name, values_: Float32(values.copy()) }
}

///|
pub fn TfliteRunnerInput::quantized(
  name : String,
  values : Array[Int],
) -> TfliteRunnerInput raise @tensor.TensorError {
  ensure_tflite_runner_name(name)
  { name_: name, values_: Quantized(values.copy()) }
}

///|
pub fn TfliteRunnerInput::name(self : TfliteRunnerInput) -> String {
  self.name_
}

///|
pub fn TfliteRunnerInput::values(
  self : TfliteRunnerInput,
) -> TfliteRunnerValues {
  match self.values_ {
    Float32(values) => Float32(values.copy())
    Quantized(values) => Quantized(values.copy())
  }
}

///|
fn tflite_runner_output(
  name : String,
  values : TfliteRunnerValues,
) -> TfliteRunnerOutput raise @tensor.TensorError {
  ensure_tflite_runner_name(name)
  { name_: name, values_: values }
}

///|
pub fn TfliteRunnerOutput::name(self : TfliteRunnerOutput) -> String {
  self.name_
}

///|
pub fn TfliteRunnerOutput::values(
  self : TfliteRunnerOutput,
) -> TfliteRunnerValues {
  match self.values_ {
    Float32(values) => Float32(values.copy())
    Quantized(values) => Quantized(values.copy())
  }
}

///|
/// Compile an arbitrary TFLite FlatBuffer for one sequential execution slot.
pub async fn TfliteRunner::compile(
  bytes : Bytes,
  preference : @compat.DevicePreference,
) -> TfliteRunner {
  TfliteRunner::compile_pool(bytes, preference, 1)
}

///|
/// Compile an arbitrary TFLite FlatBuffer with independently prepared slots.
///
/// The returned runner owns its program and must be destroyed by its owner.
pub async fn TfliteRunner::compile_pool(
  bytes : Bytes,
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
) -> TfliteRunner {
  let runtime = @litert.TfliteModel::parse_runtime(bytes)
  let builder = WebNNGraphBuilder::new(preference)
  let program = builder.compile_litert_program_pool(
    runtime.model(),
    execution_pool_size,
  )
  { program, inputs: runtime.inputs(), outputs: runtime.outputs() }
}

///|
pub fn TfliteRunner::inputs(
  self : TfliteRunner,
) -> Array[@litert.TfliteRuntimeTensor] {
  self.inputs.copy()
}

///|
pub fn TfliteRunner::outputs(
  self : TfliteRunner,
) -> Array[@litert.TfliteRuntimeTensor] {
  self.outputs.copy()
}

///|
pub async fn TfliteModelArtifact::from_bytes(
  bytes : Bytes,
) -> TfliteModelArtifact {
  let owned_bytes = Bytes::from_array(bytes.to_array())
  { bytes_: owned_bytes, digest_: tflite_model_sha256(owned_bytes).wait() }
}

///|
pub fn TfliteModelArtifact::byte_length(self : TfliteModelArtifact) -> Int {
  self.bytes_.length()
}

///|
pub fn TfliteModelArtifact::digest(self : TfliteModelArtifact) -> String {
  self.digest_
}

///|
/// Parse a TFLite FlatBuffer for synchronous CPU execution.
///
/// This runner shares the same TFLite I/O validation and quantization boundary
/// as TfliteRunner, but keeps execution independent of WebNN availability.
pub fn TfliteCpuRunner::compile(bytes : Bytes) -> TfliteCpuRunner raise {
  { runtime: @litert.TfliteModel::parse_runtime(bytes) }
}

///|
pub fn TfliteCpuRunner::inputs(
  self : TfliteCpuRunner,
) -> Array[@litert.TfliteRuntimeTensor] {
  self.runtime.inputs()
}

///|
pub fn TfliteCpuRunner::outputs(
  self : TfliteCpuRunner,
) -> Array[@litert.TfliteRuntimeTensor] {
  self.runtime.outputs()
}

///|
pub fn TfliteRunner::pool_size(self : TfliteRunner) -> Int {
  self.program.pool_size()
}

///|
fn runner_tensor_by_name(
  tensors : Array[@litert.TfliteRuntimeTensor],
  name : String,
  kind : String,
) -> @litert.TfliteRuntimeTensor raise @tensor.TensorError {
  for tensor in tensors {
    if tensor.name() == name {
      return tensor
    }
  }
  raise @tensor.TensorError::new("unknown TFLite runner \{kind} name: \{name}")
}

///|
fn ensure_runner_value_length(
  tensor : @litert.TfliteRuntimeTensor,
  value_count : Int,
) -> Unit raise @tensor.TensorError {
  let expected = tensor.shape().element_count()
  if value_count != expected {
    raise @tensor.TensorError::new(
      "TFLite runner input \{tensor.name()} value count \{value_count} does not match shape \{tensor.shape().to_string()}",
    )
  }
}

///|
fn tflite_runner_input_values(
  tensor : @litert.TfliteRuntimeTensor,
  input : TfliteRunnerInput,
) -> Array[Float] raise {
  match (tensor.data_type(), input.values_) {
    (Float32, Float32(values)) => {
      ensure_runner_value_length(tensor, values.length())
      values.copy()
    }
    (UInt8(_, _, _), Quantized(values)) => tensor.dequantize(values)
    (Int8(_, _, _), Quantized(values)) => tensor.dequantize(values)
    (Float32, Quantized(_)) =>
      raise @tensor.TensorError::new(
        "TFLite runner input \{tensor.name()} expects FLOAT32 values",
      )
    (UInt8(_, _, _), Float32(_)) =>
      raise @tensor.TensorError::new(
        "TFLite runner input \{tensor.name()} expects raw UINT8 values",
      )
    (Int8(_, _, _), Float32(_)) =>
      raise @tensor.TensorError::new(
        "TFLite runner input \{tensor.name()} expects raw INT8 values",
      )
  }
}

///|
fn tflite_runner_output_values(
  tensor : @litert.TfliteRuntimeTensor,
  values : Array[Float],
) -> TfliteRunnerValues raise {
  match tensor.data_type() {
    Float32 => Float32(values.copy())
    UInt8(_, _, _) => Quantized(tensor.requantize(values))
    Int8(_, _, _) => Quantized(tensor.requantize(values))
  }
}

///|
fn tflite_runner_bindings_by_name(
  inputs : Array[@litert.TfliteRuntimeTensor],
  input_values : Array[TfliteRunnerInput],
) -> Map[String, TfliteRunnerInput] raise @tensor.TensorError {
  if input_values.length() != inputs.length() {
    raise @tensor.TensorError::new(
      "TFLite runner input count \{input_values.length()} does not match model input count \{inputs.length()}",
    )
  }
  let expected : Map[String, Bool] = Map([])
  for input in inputs {
    expected[input.name()] = true
  }
  let bindings_by_name : Map[String, TfliteRunnerInput] = Map([])
  for binding in input_values {
    ensure_tflite_runner_name(binding.name_)
    if !expected.contains(binding.name_) {
      raise @tensor.TensorError::new(
        "unknown TFLite runner input name: \{binding.name_}",
      )
    }
    if bindings_by_name.contains(binding.name_) {
      raise @tensor.TensorError::new(
        "duplicate TFLite runner input name: \{binding.name_}",
      )
    }
    bindings_by_name[binding.name_] = binding
  }
  bindings_by_name
}

///|
/// Run named float32 or raw quantized inputs and return named TFLite outputs.
pub async fn TfliteRunner::run(
  self : TfliteRunner,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  let bindings_by_name = tflite_runner_bindings_by_name(
    self.inputs,
    input_values,
  )
  let bindings : Array[WebNNNamedValues] = []
  for tensor in self.inputs {
    let name = tensor.name()
    guard bindings_by_name.get(name) is Some(binding) else {
      raise @tensor.TensorError::new(
        "missing TFLite runner input name: \{name}",
      )
    }
    bindings.push(
      WebNNNamedValues::new(name, tflite_runner_input_values(tensor, binding)),
    )
  }
  let backend_outputs = self.program.run_named(bindings)
  let results : Array[TfliteRunnerOutput] = []
  for output in backend_outputs {
    let tensor = runner_tensor_by_name(self.outputs, output.name(), "output")
    results.push(
      tflite_runner_output(
        output.name(),
        tflite_runner_output_values(tensor, output.values()),
      ),
    )
  }
  results
}

///|
/// Run a parsed TFLite model through the CPU backend.
pub fn TfliteCpuRunner::run(
  self : TfliteCpuRunner,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] raise {
  let inputs = self.runtime.inputs()
  let outputs = self.runtime.outputs()
  let bindings_by_name = tflite_runner_bindings_by_name(inputs, input_values)
  let lowered = self.runtime
    .model()
    .lower(
      fn(name, shape) raise {
        let input_tensor = runner_tensor_by_name(inputs, name, "input")
        guard bindings_by_name.get(name) is Some(binding) else {
          raise @tensor.TensorError::new(
            "missing TFLite runner input name: \{name}",
          )
        }
        @cpu.CpuTensor::new(
          shape,
          tflite_runner_input_values(input_tensor, binding),
        )
      },
      fn(_, shape, values) raise { @cpu.CpuTensor::new(shape, values) },
    )
  let results : Array[TfliteRunnerOutput] = []
  for output in lowered {
    let tensor = runner_tensor_by_name(outputs, output.name(), "output")
    results.push(
      tflite_runner_output(
        output.name(),
        tflite_runner_output_values(tensor, output.tensor().values()),
      ),
    )
  }
  results
}

///|
pub fn TfliteRunner::destroy(self : TfliteRunner) -> Unit {
  self.program.destroy()
}

///|
pub fn TfliteRunnerCacheKey::new(
  model_digest : String,
  device : @compat.DevicePreference,
  execution_pool_size : Int,
) -> TfliteRunnerCacheKey raise @tensor.TensorError {
  if model_digest == "" {
    raise @tensor.TensorError::new(
      "TFLite runner cache model digest must not be empty",
    )
  }
  if execution_pool_size <= 0 {
    raise @tensor.TensorError::new(
      "TFLite runner execution pool size must be positive",
    )
  }
  {
    model_digest_: model_digest,
    device_: device,
    execution_pool_size_: execution_pool_size,
  }
}

///|
pub fn TfliteRunnerCacheKey::canonical(self : TfliteRunnerCacheKey) -> String {
  "tflite-runner-v1|sha256=" +
  self.model_digest_ +
  "|device=" +
  self.device_.to_webnn_string() +
  "|pool=" +
  self.execution_pool_size_.to_string()
}

///|
/// Create a cache for at most 32 models and 64 MiB of source FlatBuffers.
///
/// The byte budget is deliberately based on model bytes: WebNN does not expose
/// a portable measurement for compiled resource memory.
pub fn TfliteRunnerCache::new() -> TfliteRunnerCache {
  {
    entries: Map([]),
    entry_bytes: Map([]),
    last_access: Map([]),
    capacity_: default_tflite_runner_cache_capacity,
    byte_capacity_: default_tflite_runner_cache_byte_capacity,
    generation: { val: 0 },
    resident_bytes_: 0,
    access_counter: 0,
    hits_: 0,
    misses_: 0,
    evictions_: 0,
  }
}

///|
/// Construct a cache with a model-count limit and the default 64 MiB budget.
pub fn TfliteRunnerCache::new_with_capacity(
  capacity : Int,
) -> TfliteRunnerCache raise @tensor.TensorError {
  TfliteRunnerCache::new_with_limits(
    capacity, default_tflite_runner_cache_byte_capacity,
  )
}

///|
/// Construct a cache bounded by both entry count and source FlatBuffer bytes.
///
/// A model larger than `byte_capacity` is compiled and run transiently rather
/// than evicting the entire cache only to exceed its budget.
pub fn TfliteRunnerCache::new_with_limits(
  capacity : Int,
  byte_capacity : Int,
) -> TfliteRunnerCache raise @tensor.TensorError {
  if capacity <= 0 {
    raise @tensor.TensorError::new(
      "TFLite runner cache capacity must be positive",
    )
  }
  if byte_capacity <= 0 {
    raise @tensor.TensorError::new(
      "TFLite runner cache byte capacity must be positive",
    )
  }
  {
    entries: Map([]),
    entry_bytes: Map([]),
    last_access: Map([]),
    capacity_: capacity,
    byte_capacity_: byte_capacity,
    generation: { val: 0 },
    resident_bytes_: 0,
    access_counter: 0,
    hits_: 0,
    misses_: 0,
    evictions_: 0,
  }
}

///|
pub fn TfliteRunnerCache::capacity(self : TfliteRunnerCache) -> Int {
  self.capacity_
}

///|
pub fn TfliteRunnerCache::byte_capacity(self : TfliteRunnerCache) -> Int {
  self.byte_capacity_
}

///|
pub fn TfliteRunnerCache::resident_bytes(self : TfliteRunnerCache) -> Int {
  self.resident_bytes_
}

///|
pub fn TfliteRunnerCacheMetrics::hits(self : TfliteRunnerCacheMetrics) -> Int {
  self.hits_
}

///|
pub fn TfliteRunnerCacheMetrics::misses(self : TfliteRunnerCacheMetrics) -> Int {
  self.misses_
}

///|
pub fn TfliteRunnerCacheMetrics::evictions(
  self : TfliteRunnerCacheMetrics,
) -> Int {
  self.evictions_
}

///|
pub fn TfliteRunnerCacheMetrics::entries(
  self : TfliteRunnerCacheMetrics,
) -> Int {
  self.entries_
}

///|
pub fn TfliteRunnerCacheMetrics::resident_bytes(
  self : TfliteRunnerCacheMetrics,
) -> Int {
  self.resident_bytes_
}

///|
pub fn TfliteRunnerCacheMetrics::byte_capacity(
  self : TfliteRunnerCacheMetrics,
) -> Int {
  self.byte_capacity_
}

///|
/// Return lifetime cache counters. clear() releases programs but retains these
/// counters so callers can observe invalidation and eviction behavior.
pub fn TfliteRunnerCache::metrics(
  self : TfliteRunnerCache,
) -> TfliteRunnerCacheMetrics {
  {
    hits_: self.hits_,
    misses_: self.misses_,
    evictions_: self.evictions_,
    entries_: self.entries.length(),
    resident_bytes_: self.resident_bytes_,
    byte_capacity_: self.byte_capacity_,
  }
}

///|
fn TfliteRunnerCache::touch(
  self : TfliteRunnerCache,
  canonical : String,
) -> Unit {
  self.access_counter = self.access_counter + 1
  self.last_access[canonical] = self.access_counter
}

///|
fn TfliteRunnerCache::evict_least_recently_used(
  self : TfliteRunnerCache,
) -> Bool {
  let mut oldest_key : String? = None
  let mut oldest_access = 2147483647
  for canonical, access in self.last_access {
    if access < oldest_access {
      oldest_key = Some(canonical)
      oldest_access = access
    }
  }
  match oldest_key {
    Some(canonical) => {
      match self.entries.get(canonical) {
        Some(runner) => runner.destroy()
        None => ()
      }
      match self.entry_bytes.get(canonical) {
        Some(bytes) => self.resident_bytes_ = self.resident_bytes_ - bytes
        None => ()
      }
      self.entries.remove(canonical)
      self.entry_bytes.remove(canonical)
      self.last_access.remove(canonical)
      self.evictions_ = self.evictions_ + 1
      true
    }
    None => false
  }
}

///|
async fn TfliteRunnerCache::compile_or_get(
  self : TfliteRunnerCache,
  key : TfliteRunnerCacheKey,
  model_byte_length : Int,
  compile : () -> @js.Promise[TfliteRunner],
) -> CachedTfliteRunner raise Error {
  let canonical = key.canonical()
  match self.entries.get(canonical) {
    Some(existing) => {
      self.hits_ = self.hits_ + 1
      self.touch(canonical)
      Retained(existing)
    }
    None => {
      self.misses_ = self.misses_ + 1
      let generation = self.generation.val
      let compiled = compile().wait()
      if key.execution_pool_size_ != compiled.pool_size() {
        let actual = compiled.pool_size()
        compiled.destroy()
        raise @tensor.TensorError::new(
          "TFLite runner pool size \{actual} does not match cache key pool size \{key.execution_pool_size_}",
        )
      }
      if self.generation.val != generation {
        Transient(compiled)
      } else {
        match self.entries.get(canonical) {
          Some(existing) => {
            compiled.destroy()
            self.touch(canonical)
            Retained(existing)
          }
          None =>
            if model_byte_length > self.byte_capacity_ {
              Transient(compiled)
            } else {
              while self.entries.length() >= self.capacity_ ||
                    self.resident_bytes_ + model_byte_length >
                    self.byte_capacity_ {
                if !self.evict_least_recently_used() {
                  break
                }
              }
              self.entries[canonical] = compiled
              self.entry_bytes[canonical] = model_byte_length
              self.resident_bytes_ = self.resident_bytes_ + model_byte_length
              self.touch(canonical)
              Retained(compiled)
            }
        }
      }
    }
  }
}

///|
pub fn TfliteRunnerCache::length(self : TfliteRunnerCache) -> Int {
  self.entries.length()
}

///|
/// Clear all retained WebNN programs and prevent an already compiling runner
/// from being reinserted after this call returns.
pub fn TfliteRunnerCache::clear(self : TfliteRunnerCache) -> Int {
  let previous_length = self.entries.length()
  self.generation.val = self.generation.val + 1
  self.entries.each(fn(_, runner) { runner.destroy() })
  self.entries.clear()
  self.entry_bytes.clear()
  self.last_access.clear()
  self.resident_bytes_ = 0
  previous_length
}

///|
extern "js" fn tflite_model_sha256(bytes : Bytes) -> @js.Promise[String] =
  #| async (bytes) => {
  #|   const digest = await crypto.subtle.digest("SHA-256", bytes);
  #|   return Array.from(new Uint8Array(digest), (value) =>
  #|     value.toString(16).padStart(2, "0")
  #|   ).join("");
  #| }

///|
/// Run a model through a cache owned by self.
///
/// The SHA-256 digest covers the original FlatBuffer bytes; device preference
/// and slot count are part of the canonical cache key.
pub async fn TfliteRunnerCache::run(
  self : TfliteRunnerCache,
  bytes : Bytes,
  preference : @compat.DevicePreference,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  self.run_pool(bytes, preference, 1, input_values)
}

///|
pub async fn TfliteRunnerCache::run_pool(
  self : TfliteRunnerCache,
  bytes : Bytes,
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  let digest = tflite_model_sha256(bytes).wait()
  self.run_pool_with_digest(
    bytes, digest, preference, execution_pool_size, input_values,
  )
}

///|
pub async fn TfliteRunnerCache::run_artifact(
  self : TfliteRunnerCache,
  artifact : TfliteModelArtifact,
  preference : @compat.DevicePreference,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  self.run_artifact_pool(artifact, preference, 1, input_values)
}

///|
pub async fn TfliteRunnerCache::run_artifact_pool(
  self : TfliteRunnerCache,
  artifact : TfliteModelArtifact,
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  self.run_pool_with_digest(
    artifact.bytes_,
    artifact.digest_,
    preference,
    execution_pool_size,
    input_values,
  )
}

///|
/// Run an artifact with an instrumented compiler while preserving cache
/// ownership and invalidation semantics.
///
/// This advanced entry point is useful for compile tracing or externally
/// scheduled compilation. `compile` must return a runner for `artifact`,
/// `preference`, and `execution_pool_size`; it is evaluated only on a miss.
pub async fn TfliteRunnerCache::run_artifact_pool_with_compiler(
  self : TfliteRunnerCache,
  artifact : TfliteModelArtifact,
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  input_values : Array[TfliteRunnerInput],
  compile : () -> @js.Promise[TfliteRunner],
) -> Array[TfliteRunnerOutput] {
  let key = TfliteRunnerCacheKey::new(
    artifact.digest_,
    preference,
    execution_pool_size,
  )
  self.run_with_compiler(key, artifact.byte_length(), input_values, compile)
}

///|
async fn TfliteRunnerCache::run_pool_with_digest(
  self : TfliteRunnerCache,
  bytes : Bytes,
  digest : String,
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  let key = TfliteRunnerCacheKey::new(digest, preference, execution_pool_size)
  self.run_with_compiler(key, bytes.length(), input_values, fn() {
    @js.from_async(async fn() {
      TfliteRunner::compile_pool(bytes, preference, execution_pool_size)
    })
  })
}

///|
async fn TfliteRunnerCache::run_with_compiler(
  self : TfliteRunnerCache,
  key : TfliteRunnerCacheKey,
  model_byte_length : Int,
  input_values : Array[TfliteRunnerInput],
  compile : () -> @js.Promise[TfliteRunner],
) -> Array[TfliteRunnerOutput] {
  match self.compile_or_get(key, model_byte_length, compile) {
    Retained(runner) => runner.run(input_values)
    Transient(runner) => {
      let output = runner.run(input_values) catch {
        error => {
          runner.destroy()
          raise error
        }
      }
      runner.destroy()
      output
    }
  }
}

///|
/// Create a CPU runner cache for at most 32 parsed models and 64 MiB of source
/// FlatBuffers. It is intended for WebNNRuntime's CPU fallback path.
pub fn TfliteCpuRunnerCache::new() -> TfliteCpuRunnerCache {
  {
    entries: Map([]),
    entry_bytes: Map([]),
    last_access: Map([]),
    capacity_: default_tflite_runner_cache_capacity,
    byte_capacity_: default_tflite_runner_cache_byte_capacity,
    resident_bytes_: 0,
    access_counter: 0,
    hits_: 0,
    misses_: 0,
    evictions_: 0,
  }
}

///|
/// Construct a CPU cache with a model-count limit and the default 64 MiB
/// source-FlatBuffer budget.
pub fn TfliteCpuRunnerCache::new_with_capacity(
  capacity : Int,
) -> TfliteCpuRunnerCache raise @tensor.TensorError {
  TfliteCpuRunnerCache::new_with_limits(
    capacity, default_tflite_runner_cache_byte_capacity,
  )
}

///|
/// Construct a parsed-CPU-model cache bounded by entry count and source bytes.
pub fn TfliteCpuRunnerCache::new_with_limits(
  capacity : Int,
  byte_capacity : Int,
) -> TfliteCpuRunnerCache raise @tensor.TensorError {
  if capacity <= 0 {
    raise @tensor.TensorError::new(
      "TFLite CPU runner cache capacity must be positive",
    )
  }
  if byte_capacity <= 0 {
    raise @tensor.TensorError::new(
      "TFLite CPU runner cache byte capacity must be positive",
    )
  }
  {
    entries: Map([]),
    entry_bytes: Map([]),
    last_access: Map([]),
    capacity_: capacity,
    byte_capacity_: byte_capacity,
    resident_bytes_: 0,
    access_counter: 0,
    hits_: 0,
    misses_: 0,
    evictions_: 0,
  }
}

///|
pub fn TfliteCpuRunnerCache::capacity(self : TfliteCpuRunnerCache) -> Int {
  self.capacity_
}

///|
pub fn TfliteCpuRunnerCache::byte_capacity(self : TfliteCpuRunnerCache) -> Int {
  self.byte_capacity_
}

///|
pub fn TfliteCpuRunnerCache::resident_bytes(self : TfliteCpuRunnerCache) -> Int {
  self.resident_bytes_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::hits(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.hits_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::misses(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.misses_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::evictions(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.evictions_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::entries(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.entries_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::resident_bytes(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.resident_bytes_
}

///|
pub fn TfliteCpuRunnerCacheMetrics::byte_capacity(
  self : TfliteCpuRunnerCacheMetrics,
) -> Int {
  self.byte_capacity_
}

///|
pub fn TfliteCpuRunnerCache::metrics(
  self : TfliteCpuRunnerCache,
) -> TfliteCpuRunnerCacheMetrics {
  {
    hits_: self.hits_,
    misses_: self.misses_,
    evictions_: self.evictions_,
    entries_: self.entries.length(),
    resident_bytes_: self.resident_bytes_,
    byte_capacity_: self.byte_capacity_,
  }
}

///|
fn TfliteCpuRunnerCache::touch(
  self : TfliteCpuRunnerCache,
  canonical : String,
) -> Unit {
  self.access_counter = self.access_counter + 1
  self.last_access[canonical] = self.access_counter
}

///|
fn TfliteCpuRunnerCache::evict_least_recently_used(
  self : TfliteCpuRunnerCache,
) -> Bool {
  let mut oldest_key : String? = None
  let mut oldest_access = 2147483647
  for canonical, access in self.last_access {
    if access < oldest_access {
      oldest_key = Some(canonical)
      oldest_access = access
    }
  }
  match oldest_key {
    Some(canonical) => {
      match self.entry_bytes.get(canonical) {
        Some(bytes) => self.resident_bytes_ = self.resident_bytes_ - bytes
        None => ()
      }
      self.entries.remove(canonical)
      self.entry_bytes.remove(canonical)
      self.last_access.remove(canonical)
      self.evictions_ = self.evictions_ + 1
      true
    }
    None => false
  }
}

///|
fn TfliteCpuRunnerCache::compile_or_get(
  self : TfliteCpuRunnerCache,
  bytes : Bytes,
  digest : String,
) -> TfliteCpuRunner raise {
  match self.entries.get(digest) {
    Some(existing) => {
      self.hits_ = self.hits_ + 1
      self.touch(digest)
      existing
    }
    None => {
      self.misses_ = self.misses_ + 1
      let runner = TfliteCpuRunner::compile(bytes)
      let model_byte_length = bytes.length()
      if model_byte_length > self.byte_capacity_ {
        runner
      } else {
        while self.entries.length() >= self.capacity_ ||
              self.resident_bytes_ + model_byte_length > self.byte_capacity_ {
          if !self.evict_least_recently_used() {
            break
          }
        }
        self.entries[digest] = runner
        self.entry_bytes[digest] = model_byte_length
        self.resident_bytes_ = self.resident_bytes_ + model_byte_length
        self.touch(digest)
        runner
      }
    }
  }
}

///|
pub fn TfliteCpuRunnerCache::length(self : TfliteCpuRunnerCache) -> Int {
  self.entries.length()
}

///|
/// Release parsed CPU models while retaining lifetime hit/miss/eviction counters.
pub fn TfliteCpuRunnerCache::clear(self : TfliteCpuRunnerCache) -> Int {
  let previous_length = self.entries.length()
  self.entries.clear()
  self.entry_bytes.clear()
  self.last_access.clear()
  self.resident_bytes_ = 0
  previous_length
}

///|
/// Parse and run bytes through the CPU cache. The SHA-256 digest is computed
/// for this call; use run_artifact to reuse a digest prepared by the caller.
pub async fn TfliteCpuRunnerCache::run(
  self : TfliteCpuRunnerCache,
  bytes : Bytes,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  let digest = tflite_model_sha256(bytes).wait()
  self.compile_or_get(bytes, digest).run(input_values)
}

///|
/// Run a pre-hashed artifact through the parsed CPU-model cache.
pub fn TfliteCpuRunnerCache::run_artifact(
  self : TfliteCpuRunnerCache,
  artifact : TfliteModelArtifact,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] raise {
  self.compile_or_get(artifact.bytes_, artifact.digest_).run(input_values)
}
