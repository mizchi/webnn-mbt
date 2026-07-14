///|
pub struct WebNNProgramCacheKey {
  graph_hash_ : String
  device_ : @compat.DevicePreference
  input_shapes_ : Array[@shape.Shape]
  execution_pool_size_ : Int
}

///|
pub fn WebNNProgramCacheKey::new(
  graph_hash : String,
  device : @compat.DevicePreference,
  input_shapes : Array[@shape.Shape],
) -> WebNNProgramCacheKey raise @tensor.TensorError {
  WebNNProgramCacheKey::new_with_pool(graph_hash, device, input_shapes, 1)
}

///|
pub fn WebNNProgramCacheKey::new_with_pool(
  graph_hash : String,
  device : @compat.DevicePreference,
  input_shapes : Array[@shape.Shape],
  execution_pool_size : Int,
) -> WebNNProgramCacheKey raise @tensor.TensorError {
  if graph_hash == "" {
    raise @tensor.TensorError::new(
      "WebNN program cache graph hash must not be empty",
    )
  }
  if input_shapes.is_empty() {
    raise @tensor.TensorError::new(
      "WebNN program cache input shapes must not be empty",
    )
  }
  if execution_pool_size <= 0 {
    raise @tensor.TensorError::new(
      "WebNN program execution pool size must be positive",
    )
  }
  {
    graph_hash_: graph_hash,
    device_: device,
    input_shapes_: input_shapes.copy(),
    execution_pool_size_: execution_pool_size,
  }
}

///|
pub fn WebNNProgramCacheKey::canonical(self : WebNNProgramCacheKey) -> String {
  let input_shapes = self.input_shapes_.map(fn(shape) { shape.to_string() })
  "webnn-program-v2|graph=" +
  self.graph_hash_.length().to_string() +
  ":" +
  self.graph_hash_ +
  "|device=" +
  self.device_.to_webnn_string() +
  "|inputs=" +
  input_shapes.join(";") +
  "|pool=" +
  self.execution_pool_size_.to_string()
}

///|
pub struct WebNNProgramCache {
  entries : Map[String, WebNNProgram]
  generation : Ref[Int]
}

///|
priv enum CachedProgram {
  Retained(WebNNProgram, Bool)
  Transient(WebNNProgram)
}

///|
/// Compile a program on a cache miss and return the cache-owned instance.
///
/// This private helper is the only point where a WebNNProgram enters the
/// cache. Public methods execute the program without exposing it, so clear()
/// remains the sole public destruction path for cached resources.
async fn WebNNProgramCache::compile_or_get(
  self : WebNNProgramCache,
  key : WebNNProgramCacheKey,
  compile : () -> @js.Promise[WebNNProgram],
) -> CachedProgram raise Error {
  let canonical = key.canonical()
  match self.entries.get(canonical) {
    Some(existing) => Retained(existing, true)
    None => {
      let generation = self.generation.val
      let compiled = compile().wait()
      if key.execution_pool_size_ != compiled.pool_size() {
        let actual = compiled.pool_size()
        compiled.destroy()
        raise @tensor.TensorError::new(
          "WebNN program pool size \{actual} does not match cache key pool size \{key.execution_pool_size_}",
        )
      }
      if self.generation.val != generation {
        Transient(compiled)
      } else {
        match self.entries.get(canonical) {
          Some(existing) => {
            compiled.destroy()
            Retained(existing, true)
          }
          None => {
            self.entries[canonical] = compiled
            Retained(compiled, false)
          }
        }
      }
    }
  }
}

///|
pub fn WebNNProgramCache::new() -> WebNNProgramCache {
  { entries: Map([]), generation: { val: 0 } }
}

///|
/// Run a single-input/single-output program through this cache.
///
/// The callback is evaluated only on a cache miss. The program it returns is
/// retained and destroyed by this cache; callers receive only inference data
/// and whether the program was already cached.
pub async fn WebNNProgramCache::run_or_compile(
  self : WebNNProgramCache,
  key : WebNNProgramCacheKey,
  input_values : Array[Float],
  compile : () -> @js.Promise[WebNNProgram],
) -> (Array[Float], Bool) {
  match self.compile_or_get(key, compile) {
    Retained(program, cache_hit) => (program.run(input_values), cache_hit)
    Transient(program) => {
      let output = program.run(input_values) catch {
        error => {
          program.destroy()
          raise error
        }
      }
      program.destroy()
      (output, false)
    }
  }
}

///|
/// Run a named-I/O program through this cache.
///
/// As with run_or_compile(), this cache owns the compiled program and clear()
/// is the sole public way to release its WebNN resources.
pub async fn WebNNProgramCache::run_named_or_compile(
  self : WebNNProgramCache,
  key : WebNNProgramCacheKey,
  input_values : Array[WebNNNamedValues],
  compile : () -> @js.Promise[WebNNProgram],
) -> (Array[WebNNNamedValues], Bool) {
  match self.compile_or_get(key, compile) {
    Retained(program, cache_hit) => (program.run_named(input_values), cache_hit)
    Transient(program) => {
      let output = program.run_named(input_values) catch {
        error => {
          program.destroy()
          raise error
        }
      }
      program.destroy()
      (output, false)
    }
  }
}

///|
pub fn WebNNProgramCache::length(self : WebNNProgramCache) -> Int {
  self.entries.length()
}

///|
pub fn WebNNProgramCache::clear(self : WebNNProgramCache) -> Int {
  let previous_length = self.entries.length()
  self.generation.val = self.generation.val + 1
  self.entries.each(fn(_, program) { program.destroy() })
  self.entries.clear()
  previous_length
}
