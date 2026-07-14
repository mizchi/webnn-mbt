///|
fn supported_operators() -> @js.Promise[Array[String]] {
  @js.from_async(async fn() {
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    graph.supported_operators()
  })
}

///|
extern "js" fn make_webnn_capabilities_report(
  compatibility_mode : String,
  preferred_input_layout : String,
  op_support_limits_available : Bool,
  tensor_io_available : Bool,
  supported_operators : Array[String],
) -> @core.Any =
  #| (compatibilityMode, preferredInputLayout, opSupportLimitsAvailable, tensorIoAvailable, supportedOperators) => ({
  #|   compatibilityMode,
  #|   preferredInputLayout,
  #|   opSupportLimitsAvailable,
  #|   tensorIoAvailable,
  #|   supportedOperators,
  #| })

///|
fn probe_webnn_capabilities() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let capabilities = @webnn.probe_capabilities(@compat.DevicePreference::Npu)
    make_webnn_capabilities_report(
      capabilities.contract().to_string(),
      capabilities.preferred_input_layout().to_webnn_string(),
      capabilities.op_support_limits_available(),
      capabilities.tensor_io_available(),
      capabilities.supported_operators(),
    )
  })
}

///|
fn run_cpu_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(fn() raise {
    let input = @cpu.CpuTensor::new(@shape.Shape::new([1, 2]), [1.0, 2.0])
    let weight = @cpu.CpuTensor::new(@shape.Shape::new([2, 2]), [
      1.0, 2.0, 3.0, 4.0,
    ])
    let bias = @cpu.CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0])
    @model.Linear::{ weight, bias }.forward(input).values()
  })
}

///|
fn run_add() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let shape = @shape.Shape::new([4])
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    let input_binding = graph.input("input", shape)
    let input = input_binding.tensor()
    let bias = graph.constant(shape, [10.0, 20.0, 30.0, 40.0])
    let output = input.add(bias)
    let session = graph.compile_single(input_binding, "output", output)
    session.run([1.0, 2.0, 3.0, 4.0])
  })
}

///|
fn run_concat() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let lhs = graph.input("lhs", @shape.Shape::new([2, 2]))
    let rhs = graph.input("rhs", @shape.Shape::new([2, 1]))
    let output = lhs.tensor().concat(rhs.tensor(), 1)
    let program = graph.compile_program_named([lhs, rhs], [
      graph.output("output", output),
    ])
    defer program.destroy()
    let results = program.run_named([
      @webnn.WebNNNamedValues::new("lhs", [1.0, 2.0, 3.0, 4.0]),
      @webnn.WebNNNamedValues::new("rhs", [10.0, 20.0]),
    ])
    results[0].values()
  })
}

///|
fn run_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let input_shape = @shape.Shape::new([1, 2])
    let graph = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer graph.destroy()
    let input_binding = graph.input("input", input_shape)
    let input = input_binding.tensor()
    let weight = graph.constant(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
    let bias = graph.constant(@shape.Shape::new([2]), [1.0, 1.0])
    let linear = @model.Linear::{ weight, bias }
    let output = linear.forward(input)
    let session = graph.compile_single(input_binding, "output", output)
    session.run([1.0, 2.0])
  })
}

///|
fn run_litert_linear() -> @js.Promise[Array[Float]] {
  @js.from_async(async fn() {
    let input_shape = @shape.Shape::new([1, 2])
    let weight_shape = @shape.Shape::new([2, 2])
    let output_shape = @shape.Shape::new([1, 2])
    let model = @litert.LiteRtModel::new(
      [
        @litert.LiteRtValue::input("input", input_shape),
        @litert.LiteRtValue::constant("weight", weight_shape, [
          1.0, 2.0, 3.0, 4.0,
        ]),
        @litert.LiteRtValue::constant("bias", @shape.Shape::new([2]), [1.0, 1.0]),
        @litert.LiteRtValue::intermediate("matmul", output_shape),
        @litert.LiteRtValue::intermediate("output", output_shape),
      ],
      [
        @litert.LiteRtNode::Matmul("input", "weight", "matmul"),
        @litert.LiteRtNode::Add("matmul", "bias", "output"),
      ],
      ["output"],
    )
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    let lowered = builder.lower_litert(model)
    let input_bindings = lowered.inputs()
    let outputs = lowered.outputs()
    let program = builder.compile_program_single(
      input_bindings[0],
      outputs[0].name(),
      outputs[0].tensor(),
    )
    defer program.destroy()
    program.run([1.0, 2.0])
  })
}
