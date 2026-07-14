///|
pub(all) enum WebNNRuntimeFallback {
  Disabled
  Cpu
}

///|
pub struct WebNNRuntimeMetrics {
  cache_ : TfliteRunnerCacheMetrics
  cpu_cache_ : TfliteCpuRunnerCacheMetrics
  cpu_fallbacks_ : Int
}

///|
/// High-level, cache-owning entry point for executing TFLite models with WebNN.
///
/// A runtime fixes the device preference and execution-pool size for its
/// lifetime. It retains compiled programs internally, so users need only call
/// clear() when the runtime is no longer needed.
pub struct WebNNRuntime {
  preference_ : @compat.DevicePreference
  execution_pool_size_ : Int
  fallback_ : WebNNRuntimeFallback
  tflite_cache : TfliteRunnerCache
  cpu_tflite_cache : TfliteCpuRunnerCache
  mut cpu_fallbacks_ : Int
}

///|
pub fn WebNNRuntime::new(preference : @compat.DevicePreference) -> WebNNRuntime {
  {
    preference_: preference,
    execution_pool_size_: 1,
    fallback_: Disabled,
    tflite_cache: TfliteRunnerCache::new(),
    cpu_tflite_cache: TfliteCpuRunnerCache::new(),
    cpu_fallbacks_: 0,
  }
}

///|
pub fn WebNNRuntime::new_with_pool(
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
) -> WebNNRuntime raise @tensor.TensorError {
  if execution_pool_size <= 0 {
    raise @tensor.TensorError::new(
      "WebNN runtime execution pool size must be positive",
    )
  }
  {
    preference_: preference,
    execution_pool_size_: execution_pool_size,
    fallback_: Disabled,
    tflite_cache: TfliteRunnerCache::new(),
    cpu_tflite_cache: TfliteCpuRunnerCache::new(),
    cpu_fallbacks_: 0,
  }
}

///|
pub fn WebNNRuntime::new_with_options(
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  cache_capacity : Int,
  fallback : WebNNRuntimeFallback,
) -> WebNNRuntime raise @tensor.TensorError {
  if execution_pool_size <= 0 {
    raise @tensor.TensorError::new(
      "WebNN runtime execution pool size must be positive",
    )
  }
  if cache_capacity <= 0 {
    raise @tensor.TensorError::new(
      "WebNN runtime cache capacity must be positive",
    )
  }
  {
    preference_: preference,
    execution_pool_size_: execution_pool_size,
    fallback_: fallback,
    tflite_cache: TfliteRunnerCache::new_with_capacity(cache_capacity),
    cpu_tflite_cache: TfliteCpuRunnerCache::new_with_capacity(cache_capacity),
    cpu_fallbacks_: 0,
  }
}

///|
/// Construct a runtime with independent WebNN-program and parsed-CPU-model
/// cache budgets. Both byte limits measure source TFLite FlatBuffer bytes.
pub fn WebNNRuntime::new_with_cache_limits(
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  cache_capacity : Int,
  cache_byte_capacity : Int,
  cpu_cache_capacity : Int,
  cpu_cache_byte_capacity : Int,
  fallback : WebNNRuntimeFallback,
) -> WebNNRuntime raise @tensor.TensorError {
  if execution_pool_size <= 0 {
    raise @tensor.TensorError::new(
      "WebNN runtime execution pool size must be positive",
    )
  }
  let tflite_cache = TfliteRunnerCache::new_with_limits(
    cache_capacity, cache_byte_capacity,
  )
  let cpu_tflite_cache = TfliteCpuRunnerCache::new_with_limits(
    cpu_cache_capacity, cpu_cache_byte_capacity,
  )
  {
    preference_: preference,
    execution_pool_size_: execution_pool_size,
    fallback_: fallback,
    tflite_cache,
    cpu_tflite_cache,
    cpu_fallbacks_: 0,
  }
}

///|
/// Construct a bounded runtime that retries failed WebNN work on the CPU.
pub fn WebNNRuntime::new_with_cpu_fallback(
  preference : @compat.DevicePreference,
  execution_pool_size : Int,
  cache_capacity : Int,
) -> WebNNRuntime raise @tensor.TensorError {
  WebNNRuntime::new_with_options(
    preference,
    execution_pool_size,
    cache_capacity,
    Cpu,
  )
}

///|
pub fn WebNNRuntime::preference(
  self : WebNNRuntime,
) -> @compat.DevicePreference {
  self.preference_
}

///|
pub fn WebNNRuntime::execution_pool_size(self : WebNNRuntime) -> Int {
  self.execution_pool_size_
}

///|
pub fn WebNNRuntime::cache_size(self : WebNNRuntime) -> Int {
  self.tflite_cache.length()
}

///|
pub fn WebNNRuntime::cache_capacity(self : WebNNRuntime) -> Int {
  self.tflite_cache.capacity()
}

///|
pub fn WebNNRuntime::cache_byte_capacity(self : WebNNRuntime) -> Int {
  self.tflite_cache.byte_capacity()
}

///|
pub fn WebNNRuntime::cache_resident_bytes(self : WebNNRuntime) -> Int {
  self.tflite_cache.resident_bytes()
}

///|
pub fn WebNNRuntime::cpu_cache_size(self : WebNNRuntime) -> Int {
  self.cpu_tflite_cache.length()
}

///|
pub fn WebNNRuntime::cpu_cache_capacity(self : WebNNRuntime) -> Int {
  self.cpu_tflite_cache.capacity()
}

///|
pub fn WebNNRuntime::cpu_cache_byte_capacity(self : WebNNRuntime) -> Int {
  self.cpu_tflite_cache.byte_capacity()
}

///|
pub fn WebNNRuntime::cpu_cache_resident_bytes(self : WebNNRuntime) -> Int {
  self.cpu_tflite_cache.resident_bytes()
}

///|
pub fn WebNNRuntime::fallback(self : WebNNRuntime) -> WebNNRuntimeFallback {
  self.fallback_
}

///|
pub fn WebNNRuntimeMetrics::hits(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.hits()
}

///|
pub fn WebNNRuntimeMetrics::misses(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.misses()
}

///|
pub fn WebNNRuntimeMetrics::evictions(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.evictions()
}

///|
pub fn WebNNRuntimeMetrics::entries(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.entries()
}

///|
pub fn WebNNRuntimeMetrics::resident_bytes(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.resident_bytes()
}

///|
pub fn WebNNRuntimeMetrics::byte_capacity(self : WebNNRuntimeMetrics) -> Int {
  self.cache_.byte_capacity()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_hits(self : WebNNRuntimeMetrics) -> Int {
  self.cpu_cache_.hits()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_misses(self : WebNNRuntimeMetrics) -> Int {
  self.cpu_cache_.misses()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_evictions(
  self : WebNNRuntimeMetrics,
) -> Int {
  self.cpu_cache_.evictions()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_entries(
  self : WebNNRuntimeMetrics,
) -> Int {
  self.cpu_cache_.entries()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_resident_bytes(
  self : WebNNRuntimeMetrics,
) -> Int {
  self.cpu_cache_.resident_bytes()
}

///|
pub fn WebNNRuntimeMetrics::cpu_cache_byte_capacity(
  self : WebNNRuntimeMetrics,
) -> Int {
  self.cpu_cache_.byte_capacity()
}

///|
pub fn WebNNRuntimeMetrics::cpu_fallbacks(self : WebNNRuntimeMetrics) -> Int {
  self.cpu_fallbacks_
}

///|
pub fn WebNNRuntime::metrics(self : WebNNRuntime) -> WebNNRuntimeMetrics {
  {
    cache_: self.tflite_cache.metrics(),
    cpu_cache_: self.cpu_tflite_cache.metrics(),
    cpu_fallbacks_: self.cpu_fallbacks_,
  }
}

///|
/// Execute arbitrary TFLite FlatBuffer bytes.
///
/// Models are keyed by their SHA-256 digest, fixed device preference, and
/// fixed execution-pool size. The runtime owns every compiled program.
pub async fn WebNNRuntime::run_tflite(
  self : WebNNRuntime,
  bytes : Bytes,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  self.tflite_cache.run_pool(
    bytes,
    self.preference_,
    self.execution_pool_size_,
    input_values,
  ) catch {
    error =>
      match self.fallback_ {
        Disabled => raise error
        Cpu => {
          self.cpu_fallbacks_ = self.cpu_fallbacks_ + 1
          self.cpu_tflite_cache.run(bytes, input_values)
        }
      }
  }
}

///|
/// Execute a model whose bytes and SHA-256 key were prepared once up front.
///
/// This preserves the TFLite I/O contract of run_tflite while avoiding a
/// repeated digest over the original FlatBuffer on cache hits.
pub async fn WebNNRuntime::run_prepared_tflite(
  self : WebNNRuntime,
  artifact : TfliteModelArtifact,
  input_values : Array[TfliteRunnerInput],
) -> Array[TfliteRunnerOutput] {
  self.tflite_cache.run_artifact_pool(
    artifact,
    self.preference_,
    self.execution_pool_size_,
    input_values,
  ) catch {
    error =>
      match self.fallback_ {
        Disabled => raise error
        Cpu => {
          self.cpu_fallbacks_ = self.cpu_fallbacks_ + 1
          self.cpu_tflite_cache.run_artifact(artifact, input_values)
        }
      }
  }
}

///|
/// Clear cached WebNN programs and parsed CPU fallback models. The return value
/// remains the number of WebNN programs removed for compatibility.
pub fn WebNNRuntime::clear(self : WebNNRuntime) -> Int {
  let cleared = self.tflite_cache.clear()
  self.cpu_tflite_cache.clear() |> ignore
  cleared
}
