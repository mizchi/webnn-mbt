///|
test "TFLite runner cache key includes model digest device and pool" {
  let key = TfliteRunnerCacheKey::new(
    "0123456789abcdef",
    @compat.DevicePreference::Npu,
    2,
  )
  @test.assert_eq(
    key.canonical(),
    "tflite-runner-v1|sha256=0123456789abcdef|device=npu|pool=2",
  )
  let other_device = TfliteRunnerCacheKey::new(
    "0123456789abcdef",
    @compat.DevicePreference::Gpu,
    2,
  )
  @test.assert_eq(key.canonical() == other_device.canonical(), false)
}

///|
test "TFLite runner cache key rejects empty digest and invalid pool" {
  try TfliteRunnerCacheKey::new("", @compat.DevicePreference::Npu, 1) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "TFLite runner cache model digest must not be empty",
      )
  } noraise {
    _ => fail("expected an empty digest to be rejected")
  }
  try
    TfliteRunnerCacheKey::new("digest", @compat.DevicePreference::Npu, 0)
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "TFLite runner execution pool size must be positive",
      )
  } noraise {
    _ => fail("expected a zero pool size to be rejected")
  }
}

///|
test "TFLite runner input owns a copy of its values" {
  let source : Array[Float] = [1.0, 2.0]
  let input = TfliteRunnerInput::float32("input", source)
  source[0] = 9.0
  match input.values() {
    Float32(values) => @test.assert_eq(values, [1.0, 2.0])
    Quantized(_) => fail("expected float32 values")
  }
}

///|
test "TFLite runner cache exposes a bounded capacity and empty metrics" {
  let cache = TfliteRunnerCache::new_with_capacity(2)
  @test.assert_eq(cache.capacity(), 2)
  let metrics = cache.metrics()
  @test.assert_eq(metrics.hits(), 0)
  @test.assert_eq(metrics.misses(), 0)
  @test.assert_eq(metrics.evictions(), 0)
  @test.assert_eq(metrics.entries(), 0)
}

///|
test "TFLite runner cache exposes an entry and byte budget" {
  let cache = TfliteRunnerCache::new_with_limits(2, 1024)
  @test.assert_eq(cache.capacity(), 2)
  @test.assert_eq(cache.byte_capacity(), 1024)
  @test.assert_eq(cache.resident_bytes(), 0)
  let metrics = cache.metrics()
  @test.assert_eq(metrics.byte_capacity(), 1024)
  @test.assert_eq(metrics.resident_bytes(), 0)
}

///|
test "TFLite runner cache rejects a non-positive capacity" {
  try TfliteRunnerCache::new_with_capacity(0) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "TFLite runner cache capacity must be positive")
  } noraise {
    _ => fail("expected a zero cache capacity to be rejected")
  }
}

///|
test "TFLite runner cache rejects a non-positive byte budget" {
  try TfliteRunnerCache::new_with_limits(1, 0) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "TFLite runner cache byte capacity must be positive",
      )
  } noraise {
    _ => fail("expected a zero byte budget to be rejected")
  }
}

///|
test "CPU TFLite runner cache exposes an independent byte budget" {
  let cache = TfliteCpuRunnerCache::new_with_limits(3, 2048)
  @test.assert_eq(cache.capacity(), 3)
  @test.assert_eq(cache.byte_capacity(), 2048)
  @test.assert_eq(cache.resident_bytes(), 0)
  let metrics = cache.metrics()
  @test.assert_eq(metrics.byte_capacity(), 2048)
  @test.assert_eq(metrics.resident_bytes(), 0)
}
