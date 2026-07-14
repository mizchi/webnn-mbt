///|
test "runtime stores its selected device and pool size" {
  let runtime = WebNNRuntime::new_with_pool(@compat.DevicePreference::Npu, 2)
  @test.assert_eq(runtime.preference(), @compat.DevicePreference::Npu)
  @test.assert_eq(runtime.execution_pool_size(), 2)
  @test.assert_eq(runtime.cache_size(), 0)
}

///|
test "runtime rejects a non-positive execution pool" {
  try WebNNRuntime::new_with_pool(@compat.DevicePreference::Npu, 0) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "WebNN runtime execution pool size must be positive",
      )
  } noraise {
    _ => fail("expected a zero execution pool to be rejected")
  }
}

///|
test "runtime stores cache and fallback policy" {
  let runtime = WebNNRuntime::new_with_options(
    @compat.DevicePreference::Gpu,
    2,
    3,
    Cpu,
  )
  @test.assert_eq(runtime.cache_capacity(), 3)
  match runtime.fallback() {
    Cpu => ()
    Disabled => fail("expected CPU fallback")
  }
  let metrics = runtime.metrics()
  @test.assert_eq(metrics.hits(), 0)
  @test.assert_eq(metrics.cpu_fallbacks(), 0)
}

///|
test "runtime rejects a non-positive cache capacity" {
  try
    WebNNRuntime::new_with_options(
      @compat.DevicePreference::Npu,
      1,
      0,
      Disabled,
    )
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "WebNN runtime cache capacity must be positive")
  } noraise {
    _ => fail("expected a zero cache capacity to be rejected")
  }
}

///|
test "runtime keeps independent WebNN and CPU cache byte budgets" {
  let runtime = WebNNRuntime::new_with_cache_limits(
    @compat.DevicePreference::Npu,
    2,
    3,
    4096,
    4,
    8192,
    Cpu,
  )
  @test.assert_eq(runtime.cache_capacity(), 3)
  @test.assert_eq(runtime.cache_byte_capacity(), 4096)
  @test.assert_eq(runtime.cpu_cache_capacity(), 4)
  @test.assert_eq(runtime.cpu_cache_byte_capacity(), 8192)
  let metrics = runtime.metrics()
  @test.assert_eq(metrics.resident_bytes(), 0)
  @test.assert_eq(metrics.cpu_cache_resident_bytes(), 0)
}
