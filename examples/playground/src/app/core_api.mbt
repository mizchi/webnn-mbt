///|
fn run_grouped_conv() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_binding = builder.input("input", @shape.Shape::new([1, 2, 2, 2]))
    let filter = builder.constant(@shape.Shape::new([2, 1, 1, 1]), [2.0, 3.0])
    let options = @shape.Conv2dOptions::new([0, 0, 0, 0], [1, 1], [1, 1], 2)
    let output = input_binding.tensor().conv2d(filter, options)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  })
}

///|
extern "js" fn make_preferred_layout_conv_report(
  preferred_input_layout : String,
  input_shape : Array[Int],
  filter_shape : Array[Int],
  output : Array[Float],
) -> @core.Any =
  #| (preferredInputLayout, inputShape, filterShape, output) => ({
  #|   preferredInputLayout,
  #|   inputShape,
  #|   filterShape,
  #|   output,
  #| })

///|
fn run_preferred_layout_conv() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_layout = builder.preferred_input_layout()
    let (input_shape, filter_shape, filter_layout) = match input_layout {
      @shape.InputLayout::Nchw =>
        (
          @shape.Shape::new([1, 1, 3, 3]),
          @shape.Shape::new([1, 1, 2, 2]),
          @shape.Conv2dFilterLayout::Oihw,
        )
      @shape.InputLayout::Nhwc =>
        (
          @shape.Shape::new([1, 3, 3, 1]),
          @shape.Shape::new([2, 2, 1, 1]),
          @shape.Conv2dFilterLayout::Hwio,
        )
    }
    let input_binding = builder.input("input", input_shape)
    let filter = builder.constant(filter_shape, [1.0, 0.0, 0.0, -1.0])
    let options = @shape.Conv2dOptions::new_with_layout(
      [0, 0, 0, 0],
      [1, 1],
      [1, 1],
      1,
      input_layout,
      filter_layout,
    )
    let output = input_binding.tensor().conv2d(filter, options)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    make_preferred_layout_conv_report(
      input_layout.to_webnn_string(),
      input_shape.dimensions(),
      filter_shape.dimensions(),
      program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )
  })
}

///|
fn benchmark_tiny_cnn(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.tiny_cnn(batch_size, warmup, iterations) })
}

///|
fn benchmark_tiny_cnn_layout(
  input_layout : String,
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let layout = match input_layout {
      "nchw" => @shape.InputLayout::Nchw
      "nhwc" => @shape.InputLayout::Nhwc
      _ =>
        raise @tensor.TensorError::new(
          "Tiny CNN input layout must be nchw or nhwc",
        )
    }
    @bench.tiny_cnn_with_layout(batch_size, warmup, iterations, layout)
  })
}

///|
fn run_conv_softmax() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let input_binding = builder.input("input", @shape.Shape::new([1, 1, 3, 3]))
    let filter = builder.constant(@shape.Shape::new([1, 1, 2, 2]), [
      1.0, 0.0, 0.0, 0.0,
    ])
    let output = input_binding
      .tensor()
      .conv2d(filter, @shape.Conv2dOptions::default())
      .softmax(3)
    let program = builder.compile_program_single(
      input_binding, "output", output,
    )
    defer program.destroy()
    program.run([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  })
}

///|
fn benchmark_named_io(
  size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.named_io(size, warmup, iterations) })
}

///|
extern "js" fn make_named_multi_io_report(
  names : Array[String],
  values : Array[Array[Float]],
  duplicate_input_rejected : Bool,
  unknown_input_rejected : Bool,
  input_count_rejected : Bool,
  input_length_rejected : Bool,
  duplicate_output_rejected : Bool,
) -> @core.Any =
  #| (names, values, duplicateInputRejected, unknownInputRejected, inputCountRejected, inputLengthRejected, duplicateOutputRejected) => ({
  #|   names,
  #|   outputs: Object.fromEntries(
  #|     names.map((name, index) => [name, values[index]]),
  #|   ),
  #|   duplicateInputRejected,
  #|   unknownInputRejected,
  #|   inputCountRejected,
  #|   inputLengthRejected,
  #|   duplicateOutputRejected,
  #| })

///|
fn run_named_multi_io() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let shape = @shape.Shape::new([2])
    let x = builder.input("x", shape)
    let y = builder.input("y", shape)
    let product = x.tensor().mul(y.tensor())
    let sum = x.tensor().add(y.tensor())
    let duplicate_output_rejected = try
      builder.compile_named([x, y], [
        builder.output("duplicate", product),
        builder.output("duplicate", sum),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let program = builder.compile_program_pool_named(
      [x, y],
      [builder.output("product", product), builder.output("sum", sum)],
      2,
    )
    defer program.destroy()
    let duplicate_input_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
        @webnn.WebNNNamedValues::new("x", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let unknown_input_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
        @webnn.WebNNNamedValues::new("z", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let input_count_rejected = try
      program.run_named([@webnn.WebNNNamedValues::new("x", [1.0, 2.0])])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let input_length_rejected = try
      program.run_named([
        @webnn.WebNNNamedValues::new("x", [1.0]),
        @webnn.WebNNNamedValues::new("y", [3.0, 4.0]),
      ])
    catch {
      _ => true
    } noraise {
      _ => false
    }
    let results = program.run_named([
      @webnn.WebNNNamedValues::new("y", [3.0, 4.0]),
      @webnn.WebNNNamedValues::new("x", [1.0, 2.0]),
    ])
    make_named_multi_io_report(
      results.map(fn(result) { result.name() }),
      results.map(fn(result) { result.values() }),
      duplicate_input_rejected,
      unknown_input_rejected,
      input_count_rejected,
      input_length_rejected,
      duplicate_output_rejected,
    )
  })
}

///|
fn benchmark_mnist_pool(
  pool_size : Int,
  request_count : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @mnist.benchmark_pool(pool_size, request_count, warmup, iterations)
  })
}
