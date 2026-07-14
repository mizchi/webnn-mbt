///|
test "program cache key includes graph, device, and ordered input shapes" {
  let key = WebNNProgramCacheKey::new(
    "mnist-weights-v1",
    @compat.DevicePreference::Npu,
    [@shape.Shape::new([1, 784]), @shape.Shape::new([128])],
  )
  @test.assert_eq(
    key.canonical(),
    "webnn-program-v2|graph=16:mnist-weights-v1|device=npu|inputs=[1, 784];[128]|pool=1",
  )
  let different_shape = WebNNProgramCacheKey::new(
    "mnist-weights-v1",
    @compat.DevicePreference::Npu,
    [@shape.Shape::new([16, 784]), @shape.Shape::new([128])],
  )
  @test.assert_eq(key.canonical() == different_shape.canonical(), false)
  let pooled = WebNNProgramCacheKey::new_with_pool(
    "mnist-weights-v1",
    @compat.DevicePreference::Npu,
    [@shape.Shape::new([1, 784]), @shape.Shape::new([128])],
    2,
  )
  @test.assert_eq(key.canonical() == pooled.canonical(), false)
}

///|
test "program cache key rejects an empty graph identity" {
  try
    WebNNProgramCacheKey::new("", @compat.DevicePreference::Npu, [
      @shape.Shape::new([1, 784]),
    ])
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "WebNN program cache graph hash must not be empty",
      )
    _ => fail("expected TensorError")
  } noraise {
    _ => fail("expected an empty graph hash to be rejected")
  }
}

///|
test "program cache key rejects a non-positive execution pool" {
  try
    WebNNProgramCacheKey::new_with_pool(
      "mnist-weights-v1",
      @compat.DevicePreference::Npu,
      [@shape.Shape::new([1, 784])],
      0,
    )
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "WebNN program execution pool size must be positive",
      )
    _ => fail("expected TensorError")
  } noraise {
    _ => fail("expected a zero execution pool to be rejected")
  }
}
