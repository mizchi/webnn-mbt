///|
test "shape validates dimensions and counts elements" {
  let shape = Shape::new([2, 3])
  @test.assert_eq(shape.dimensions(), [2, 3])
  @test.assert_eq(shape.element_count(), 6)
}

///|
test "shape infers broadcast dimensions" {
  let lhs = Shape::new([2, 1, 3])
  let rhs = Shape::new([1, 4, 3])
  @test.assert_eq(Shape::broadcast(lhs, rhs).dimensions(), [2, 4, 3])
}

///|
test "shape infers matrix multiplication dimensions" {
  let lhs = Shape::new([2, 3])
  let rhs = Shape::new([3, 4])
  @test.assert_eq(Shape::matmul(lhs, rhs).dimensions(), [2, 4])
}

///|
test "shape infers broadcast batched matrix multiplication dimensions" {
  let lhs = Shape::new([2, 1, 3, 4])
  let rhs = Shape::new([1, 7, 4, 5])
  @test.assert_eq(Shape::matmul(lhs, rhs).dimensions(), [2, 7, 3, 5])
  @test.assert_eq(
    Shape::matmul(Shape::new([2, 3, 4]), Shape::new([4, 5])).dimensions(),
    [2, 3, 5],
  )
}

///|
test "shape rejects incompatible matmul batch dimensions" {
  try Shape::matmul(Shape::new([2, 3, 4]), Shape::new([5, 4, 6])) catch {
    error => @test.assert_eq(error.to_string(), "cannot broadcast [2] and [5]")
  } noraise {
    _ => fail("expected incompatible matmul batch dimensions to be rejected")
  }
}

///|
test "shape infers binary concatenation dimensions" {
  let lhs = Shape::new([2, 3, 4])
  let rhs = Shape::new([2, 5, 4])
  @test.assert_eq(Shape::concat(lhs, rhs, 1).dimensions(), [2, 8, 4])
}

///|
test "shape rejects concatenation with incompatible non-axis dimensions" {
  try Shape::concat(Shape::new([2, 3]), Shape::new([3, 4]), 1) catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "cannot concatenate [2, 3] and [3, 4] along axis 1",
      )
  } noraise {
    _ => fail("expected incompatible concat shapes to be rejected")
  }
}

///|
test "shape infers transpose dimensions" {
  let shape = Shape::new([2, 3, 4])
  @test.assert_eq(shape.transpose([2, 0, 1]).dimensions(), [4, 2, 3])
}

///|
test "shape validates layer normalization parameters in axis order" {
  let input = Shape::new([3, 2, 4])
  let parameter = Shape::new([4, 3])
  let output = Shape::layer_normalization(
    input,
    parameter,
    parameter,
    [2, 0],
    1.0e-5,
  )
  @test.assert_eq(output.dimensions(), [3, 2, 4])
}

///|
test "shape rejects mismatched layer normalization parameters" {
  try
    Shape::layer_normalization(
      Shape::new([2, 3]),
      Shape::new([2]),
      Shape::new([3]),
      [1],
      1.0e-5,
    )
  catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "layer normalization scale shape [2] must match axes shape [3]",
      )
  } noraise {
    _ => fail("expected mismatched normalization scale to be rejected")
  }
}

///|
test "shape rejects invalid layer normalization axes and epsilon" {
  let input = Shape::new([2, 3])
  let parameter = Shape::new([3])
  try
    Shape::layer_normalization(
      input,
      Shape::new([3, 3]),
      Shape::new([3, 3]),
      [1, 1],
      1.0e-5,
    )
  catch {
    error =>
      @test.assert_eq(error.to_string(), "invalid layer normalization axis 1")
  } noraise {
    _ => fail("expected duplicate normalization axes to be rejected")
  }
  try Shape::layer_normalization(input, parameter, parameter, [1], 0.0) catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "layer normalization epsilon must be positive",
      )
  } noraise {
    _ => fail("expected zero normalization epsilon to be rejected")
  }
}

///|
test "shape rejects non-positive dimensions" {
  try Shape::new([2, 0]) catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "dimension 1 must be positive, received 0",
      )
  } noraise {
    _ => fail("expected Shape::new to reject a zero dimension")
  }
}

///|
test "shape rejects incompatible matmul dimensions" {
  let lhs = Shape::new([2, 3])
  let rhs = Shape::new([4, 2])
  try Shape::matmul(lhs, rhs) catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "matmul inner dimensions differ: 3 and 4",
      )
  } noraise {
    _ => fail("expected Shape::matmul to reject incompatible dimensions")
  }
}

///|
test "shape infers grouped NCHW/OIHW conv2d dimensions" {
  let options = Conv2dOptions::new([1, 1, 0, 0], [2, 1], [1, 2], 2)
  let output = Shape::conv2d_nchw_oihw(
    Shape::new([2, 4, 7, 8]),
    Shape::new([6, 2, 3, 2]),
    options,
  )
  @test.assert_eq(output.dimensions(), [2, 6, 4, 6])
}

///|
test "conv2d options reject invalid strides" {
  try Conv2dOptions::new([0, 0, 0, 0], [0, 1], [1, 1], 1) catch {
    error =>
      @test.assert_eq(error.to_string(), "conv2d strides must be positive")
  } noraise {
    _ => fail("expected a zero conv2d stride to be rejected")
  }
}

///|
test "shape infers grouped NHWC/HWIO conv2d dimensions" {
  let options = Conv2dOptions::new_with_layout(
    [1, 1, 0, 0],
    [2, 1],
    [1, 2],
    2,
    InputLayout::Nhwc,
    Conv2dFilterLayout::Hwio,
  )
  let output = Shape::conv2d(
    Shape::new([2, 7, 8, 4]),
    Shape::new([3, 2, 2, 6]),
    options,
  )
  @test.assert_eq(output.dimensions(), [2, 4, 6, 6])
}

///|
test "shape infers NCHW pool2d dimensions" {
  let options = Pool2dOptions::new([2, 2], [0, 0, 0, 0], [1, 1], [1, 1])
  let output = Shape::pool2d(Shape::new([1, 1, 3, 3]), options)
  @test.assert_eq(output.dimensions(), [1, 1, 2, 2])
}
