///|
test "LiteRT IR lowers a constant linear graph through TensorOps" {
  let input_shape = @shape.Shape::new([1, 2])
  let weight_shape = @shape.Shape::new([2, 2])
  let output_shape = @shape.Shape::new([1, 2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("weight", weight_shape, [1.0, 2.0, 3.0, 4.0]),
      LiteRtValue::constant("bias", @shape.Shape::new([2]), [1.0, 1.0]),
      LiteRtValue::intermediate("matmul", output_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [
      LiteRtNode::Matmul("input", "weight", "matmul"),
      LiteRtNode::Add("matmul", "bias", "output"),
    ],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].name(), "output")
  @test.assert_eq(outputs[0].tensor().values(), [8.0, 11.0])
}

///|
test "LiteRT IR rejects a node that overwrites an input" {
  let shape = @shape.Shape::new([1])
  let rejected = try {
    let _ = LiteRtModel::new(
      [LiteRtValue::input("input", shape)],
      [LiteRtNode::Relu("input", "input")],
      ["input"],
    )
    false
  } catch {
    _ => true
  }
  @test.assert_eq(rejected, true)
}

///|
test "LiteRT IR lowers mul reshape and transpose through TensorOps" {
  let input_shape = @shape.Shape::new([2, 2])
  let reshaped_shape = @shape.Shape::new([4, 1])
  let output_shape = @shape.Shape::new([1, 4])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("multiplier", input_shape, [2.0, 3.0, 4.0, 5.0]),
      LiteRtValue::intermediate("scaled", input_shape),
      LiteRtValue::intermediate("reshaped", reshaped_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [
      LiteRtNode::Mul("input", "multiplier", "scaled"),
      LiteRtNode::Reshape("scaled", "reshaped"),
      LiteRtNode::Transpose("reshaped", "output", [1, 0]),
    ],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 6.0, 12.0, 20.0])
}

///|
test "LiteRT IR lowers broadcast SUB through TensorOps" {
  let input_shape = @shape.Shape::new([2, 2])
  let bias_shape = @shape.Shape::new([2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("bias", bias_shape, [1.0, 10.0]),
      LiteRtValue::intermediate("output", input_shape),
    ],
    [LiteRtNode::Sub("input", "bias", "output")],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [3.0, 20.0, 5.0, 40.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 10.0, 4.0, 30.0])
}

///|
test "LiteRT IR lowers broadcast DIV through TensorOps" {
  let input_shape = @shape.Shape::new([2, 2])
  let divisor_shape = @shape.Shape::new([2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("divisor", divisor_shape, [2.0, 4.0]),
      LiteRtValue::intermediate("output", input_shape),
    ],
    [LiteRtNode::Div("input", "divisor", "output")],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [8.0, 20.0, 12.0, 40.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [4.0, 5.0, 6.0, 10.0])
}

///|
test "LiteRT IR lowers keep-dim MEAN through TensorOps" {
  let input_shape = @shape.Shape::new([2, 2])
  let output_shape = @shape.Shape::new([2, 1])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [LiteRtNode::ReduceMean("input", "output", [1], true)],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 3.0, 5.0, 7.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 6.0])
}

///|
test "LiteRT IR lowers constant-index GATHER through TensorOps" {
  let input_shape = @shape.Shape::new([3, 2])
  let indices_shape = @shape.Shape::new([2])
  let output_shape = @shape.Shape::new([2, 2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [LiteRtNode::Gather("input", "output", [2, 0], indices_shape, 0)],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [5.0, 6.0, 1.0, 2.0])
}

///|
test "LiteRT IR lowers constant-control SLICE through TensorOps" {
  let input_shape = @shape.Shape::new([2, 3])
  let output_shape = @shape.Shape::new([1, 2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [LiteRtNode::Slice("input", "output", [1, 1], [1, 2])],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [5.0, 6.0])
}

///|
test "LiteRT IR lowers layer normalization through TensorOps" {
  let input_shape = @shape.Shape::new([2, 2])
  let parameter_shape = @shape.Shape::new([2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("scale", parameter_shape, [1.0, 2.0]),
      LiteRtValue::constant("bias", parameter_shape, [0.5, -0.5]),
      LiteRtValue::intermediate("output", input_shape),
    ],
    [
      LiteRtNode::LayerNormalization(
        "input",
        "scale",
        "bias",
        "output",
        [1],
        1.0e-5,
      ),
    ],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 3.0, 5.0, 7.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  let values = outputs[0].tensor().values()
  @test.assert_eq((values[0] + 0.5).abs() < 1.0e-4, true)
  @test.assert_eq((values[1] - 1.5).abs() < 1.0e-4, true)
  @test.assert_eq((values[2] + 0.5).abs() < 1.0e-4, true)
  @test.assert_eq((values[3] - 1.5).abs() < 1.0e-4, true)
}

///|
test "LiteRT IR lowers binary concatenation through TensorOps" {
  let lhs_shape = @shape.Shape::new([2, 2])
  let rhs_shape = @shape.Shape::new([2, 1])
  let output_shape = @shape.Shape::new([2, 3])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("lhs", lhs_shape),
      LiteRtValue::input("rhs", rhs_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [LiteRtNode::Concat("lhs", "rhs", "output", 1)],
    ["output"],
  )
  let outputs = model.lower(
    (name, shape) => {
      if name == "lhs" {
        @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0])
      } else {
        @cpu.CpuTensor::new(shape, [10.0, 20.0])
      }
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [1.0, 2.0, 10.0, 3.0, 4.0, 20.0])
}

///|
test "LiteRT IR lowers conv2d relu and softmax through TensorOps" {
  let input_shape = @shape.Shape::new([1, 1, 3, 3])
  let filter_shape = @shape.Shape::new([1, 1, 2, 2])
  let output_shape = @shape.Shape::new([1, 1, 2, 2])
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", input_shape),
      LiteRtValue::constant("filter", filter_shape, [1.0, 0.0, 0.0, 0.0]),
      LiteRtValue::intermediate("conv", output_shape),
      LiteRtValue::intermediate("relu", output_shape),
      LiteRtValue::intermediate("output", output_shape),
    ],
    [
      LiteRtNode::Conv2d(
        "input",
        "filter",
        "conv",
        @shape.Conv2dOptions::default(),
      ),
      LiteRtNode::Relu("conv", "relu"),
      LiteRtNode::Softmax("relu", "output", 3),
    ],
    ["output"],
  )
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  let values = outputs[0].tensor().values()
  @test.assert_eq(values.length(), 4)
  @test.assert_eq(values[0] < values[1], true)
  @test.assert_eq(values[2] < values[3], true)
}

///|
test "TFLite fused activation lowers RELU6 after its primary operation" {
  let shape = @shape.Shape::new([3])
  let output : TfliteTensor = {
    name: "output",
    shape,
    tensor_type: 0,
    buffer_index: 0,
    quantization: None,
  }
  let fused = fused_activation(3, "pre_activation", output)
  let model = LiteRtModel::new(
    [
      LiteRtValue::input("input", shape),
      LiteRtValue::intermediate("pre_activation", shape),
      LiteRtValue::intermediate("output", shape),
    ],
    [LiteRtNode::Relu("input", "pre_activation"), ..fused.nodes],
    ["output"],
  )
  let outputs = model.lower(
    (_name, input_shape) => @cpu.CpuTensor::new(input_shape, [-2.0, 0.5, 8.0]),
    (_name, input_shape, values) => @cpu.CpuTensor::new(input_shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [0.0, 0.5, 6.0])
}

///|
test "TFLite parser rejects a non-TFL3 FlatBuffer" {
  let rejected = try {
    let _ = TfliteModel::parse(b"not-a-tflite-model")
    false
  } catch {
    _ => true
  }
  @test.assert_eq(rejected, true)
}

///|
test "TFLite tensor names fall back deterministically when names collide" {
  let shape = @shape.Shape::new([1])
  let tensors : Array[TfliteTensor] = [
    {
      name: "input",
      shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "input",
      shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "tensor_2",
      shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "tensor_2",
      shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
  ]
  let unique = unique_tflite_tensor_names(tensors)
  @test.assert_eq(unique.map(fn(tensor) { tensor.name }), [
    "input", "tensor_1", "tensor_2", "tensor_3",
  ])
}

///|
extern "js" fn tflite_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAGAAAACwBAAABAAAACAAAAAQABAAEAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAALgAAAC8AAAAwAAAAAMAAAAYAAAASAAAAHgAAAAMABAABAAAAAgADAAMAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAEAAAAAIAAAABAAAAAgAAAAYAAAB3ZWlnaHQAAAwAEAAEAAAACAAMAAwAAAAMAAAAAAAAABAAAAACAAAAAQAAAAIAAAAGAAAAb3V0cHV0AAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKAAwAAAAEAAgAAAAMAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAIAAAAMAAAAFAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAAAgAAAAAAOBAAAAQQQ=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_two_input_two_output_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAAWAAAAAAAAACcAQAAAgAAABQAAAAwAAAADAAUAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAEwAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACwAAAAuAAAAMAAAAAEAAAAHAAAADwAAABcAAAAfAAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAoAEAAEAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAABAAAAAgAAAAIAAAAAAAAAAQAAAAIAAAACAAAAAwAAAAIAAAAUAAAAQAAAAAoAEAAAAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAoAEAAEAAgADAAAAAwAAAABAAAACAAAAAwAAAABAAAAAAAAAAEAAAADAAAAAQAAAAwAAAAGAAgAAAAAAAgAAAAAAAAA"),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_sub_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAcAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAKQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACIAAAAkAAAAJQAAAADAAAAGAAAADgAAABYAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAACgAQAAQAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAEAAAACAAAAAgAAAAAAAAABAAAAAQAAAAIAAAABAAAAEAAAAAoAEAAAAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAMAAAABgAIAAAAAAAIAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_add_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAGAAAACwBAAABAAAACAAAAAQABAAEAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAALgAAAC8AAAAwAAAAAMAAAAYAAAASAAAAHgAAAAMABAABAAAAAgADAAMAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAEAAAAAIAAAABAAAAAgAAAAYAAAB3ZWlnaHQAAAwAEAAEAAAACAAMAAwAAAAMAAAAAAAAABAAAAACAAAAAQAAAAIAAAAGAAAAb3V0cHV0AAABAAAAAAAAAAEAAAACAAAAAQAAAHAAAAAKAAwAAAAEAAgAAAAMAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAIAAAAMAAAAFAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAAAgAAAAAAOBAAAAQQQ4AFAAAAAQACAAMABAAAAAQAAAAEAAAABgAAAALAAAAIAAAAAIAAAAAAAAAAQAAAAEAAAACAAAABgAIAAQAAAAIAAAAAwAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_fully_connected_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAAHABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAAJAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAOwAAADwAAAA9AAAAAQAAAAcAAAATAAAAHwAAACoAAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAEAAAAAIAAAABAAAAAgAAAAUAAABpbnB1dAAMABAABAAAAAgADAAAAA4AAAAMAAAAAQAAABAAAAACAAAAAgAAAAIAAAAHAAAAd2VpZ2h0cwAMABAABAAAAAgADAAMAAAADAAAAAIAAAAMAAAAAQAAAAIAAAAEAAAAYmlhcwAADAAQAAQAAAAIAAwAAAAOAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAAAQAAAACgAMAAAABAAIAAAADAAAAAgAAAAUAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAAwAAABAAAAAYAAAAOAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAABAAAAAAAIA/AABAQAAAAEAAAIBABgAIAAQAAAAIAAAABAAAAAgAAAAAAIA/AACAPw=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_fully_connected_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAAHABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAAJAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAOwAAADwAAAA9AAAAAQAAAAcAAAATAAAAHwAAACoAAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAEAAAAAIAAAABAAAAAgAAAAUAAABpbnB1dAAMABAABAAAAAgADAAAAA4AAAAMAAAAAQAAABAAAAACAAAAAgAAAAIAAAAHAAAAd2VpZ2h0cwAMABAABAAAAAgADAAMAAAADAAAAAIAAAAMAAAAAQAAAAIAAAAEAAAAYmlhcwAADAAQAAQAAAAIAAwAAAAOAAAADAAAAAAAAAAQAAAAAgAAAAEAAAACAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAACcAAAACgAMAAAABAAIAAAADAAAAAgAAAAUAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAAwAAABAAAAAYAAAAOAAAAAQABAAEAAAABgAIAAQAAAAIAAAABAAAABAAAAAAAIA/AABAQAAAAEAAAIBABgAIAAQAAAAIAAAABAAAAAgAAAAAAIA/AACAPw4AFAAAAAQACAAMABAAAAAQAAAAEAAAABwAAAAIAAAAJAAAAAMAAAAAAAAAAQAAAAIAAAABAAAAAwAAAAYACAAEAAAACAAAAAMAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tensorflow_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("JAAAAFRGTDMAAAAAAAAAABQAGAAEAAgADAAAABAAAAAAABQAFAAAAAMAAADkAQAAmAAAAIAAAAAEAAAAAQAAABAAAAAAAAoAEAAEAAgADAAKAAAAPAAAABwAAAAEAAAADwAAAHNlcnZpbmdfZGVmYXVsdAABAAAABAAAAOT///8IAAAAAgAAAAEAAAB4AAAAAQAAAAwAAAAIAAwABAAIAAgAAAAIAAAAAQAAAAEAAABhAAAAAQAAAAQAAACk/v//AAAAAAAAAAABAAAAEAAAAAwAFAAEAAgADAAQAAwAAACUAAAAiAAAAHwAAAAEAAAAAgAAAEQAAAAEAAAA0v///wAAAAsYAAAADAAAAAQAAAD4/v//AQAAAAIAAAACAAAAAAAAAAEAAAAAAA4AFAAAAAgADAAHABAADgAAAAAAAAsYAAAADAAAAAQAAAA0////AQAAAAAAAAACAAAAAQAAAAEAAAABAAAAAgAAAAEAAAABAAAAAwAAAHAAAAA0AAAABAAAAKj///8UAAAABAAAAAYAAABvdXRwdXQAAAQAAAABAAAACAAAAAgAAAADAAAA1P///xQAAAAEAAAABQAAAGlucHV0AAAABAAAAAEAAAAIAAAACAAAAAMAAAAMAAwABAAAAAAACAAMAAAAEAAAAAQAAAADAAAAYWRkAAQAAAABAAAACAAAAAgAAAADAAAAAQAAAAgAAAAEAAQABAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_conv2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABQABAAIAAwAAAAQAAAAEAAAAAMAAAAMAAAAJAAAALABAAABAAAAEAAAAAwACAAAAAAAAAAEAAwAAAADAAAAAQAAABAAAAAMABQABAAIAAwAEAAMAAAAEAAAAAQBAAAIAQAADAEAAAQAAAAcAAAAVAAAAIwAAAC4AAAADAAQAAQAAAAIAAwADAAAAAwAAAAAAAAAGAAAAAQAAAABAAAAAwAAAAMAAAABAAAABQAAAGlucHV0AAwAEAAEAAAACAAMAAAADgAAAAwAAAABAAAAGAAAAAQAAAABAAAAAgAAAAIAAAABAAAABgAAAGZpbHRlcgAADAAQAAQAAAAIAAwADAAAAAwAAAACAAAADAAAAAEAAAABAAAABAAAAGJpYXMAAAwAEAAEAAAACAAMAAAADgAAAAwAAAAAAAAAGAAAAAQAAAABAAAAAgAAAAIAAAABAAAABgAAAG91dHB1dAAAAQAAAAAAAAABAAAAAwAAAAEAAAAUAAAADgAUAAAABAAIAAwAEAAAABAAAAAQAAAAHAAAAAEAAAAoAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAACgAQAAQACAAMAAAADAAAAAEAAAABAAAAAQAAAAMAAAAQAAAAGAAAADgAAAAEAAQABAAAAAYACAAEAAAACAAAAAQAAAAQAAAAAACAPwAAAAAAAAAAAACAvwYACAAEAAAACAAAAAQAAAAEAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_reshape_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAsAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFgAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACcAAAAoAAAAKQAAAADAAAAGAAAAEAAAABkAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAgAAAAIAAAAMABQABAAIAAwAAAAMAAAAEAAAAAIAAAABAAAAAAAAAAEAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAABAAAAAEAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAgAAAAEAAAAAQAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_transpose_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAsAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAJwAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACcAAAAoAAAAKQAAAADAAAAGAAAAEAAAABkAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAgAAAAMAAAAMABQABAAIAAwAAAAMAAAAEAAAAAIAAAABAAAAAAAAAAEAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAACAAAAAwAAAAIAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAgAAAABAAAAAAAAAA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_depthwise_conv2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAC4AQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAABAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAADkAAAA6AAAAOwAAAAEAAAAHAAAAEwAAAB8AAAAoAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAADAAAAAwAAAAIAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAABAAAAAAAAAAQAAAABAAAAAgAAAAIAAAACAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAgAAAAAAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAABAAAAAEAAAACAAAAAgAAAAIAAAABAAAAAAAAAAEAAAADAAAAAQAAABQAAAAOABgABAAIAAwAAAAUAAAAEAAAAAAAAAAQAAAAHAAAAAAAAAAwAAAAAwAAAAAAAAABAAAAAgAAAAEAAAADAAAAEgAgAAQACAAMABAAFAAYABwAAAAUAAAAAQAAAAEAAAABAAAAAQAAAAAAAAABAAAAAQAAAAMAAAAUAAAAIAAAAFAAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAACAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEAAAOBAAAAAQQYACAAEAAAACAAAAAQAAAAIAAAAAACAPwAAIME="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_average_pool2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAABEAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAAAAhAAAAIgAAAACAAAAFAAAAEQAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAQAAAABAAAAAwAAAAMAAAABAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAAEAAAAAQAAAAIAAAACAAAAAQAAAAEAAAAAAAAAAQAAAAEAAAABAAAAFAAAAA4AGAAEAAgADAAAABQAAAAQAAAAAAAAABAAAAAUAAAAAAAAACQAAAABAAAAAAAAAAEAAAABAAAAEAAcAAQACAAMABAAFAAYABAAAAABAAAAAQAAAAEAAAACAAAAAgAAAAAAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_max_pool2d_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAABEAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAEQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAAAAhAAAAIgAAAACAAAAFAAAAEQAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAQAAAABAAAAAwAAAAMAAAABAAAADAAUAAQACAAMAAAADAAAABAAAAAAAAAAAAAAAAAAAAAEAAAAAQAAAAIAAAACAAAAAQAAAAEAAAAAAAAAAQAAAAEAAAABAAAAFAAAAA4AGAAEAAgADAAAABQAAAAQAAAAAAAAABAAAAAUAAAAAAAAACQAAAABAAAAAAAAAAEAAAABAAAAEAAcAAQACAAMABAAFAAYABAAAAABAAAAAQAAAAEAAAACAAAAAgAAAAAAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_logistic_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAADgAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAACAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_tanh_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAHAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAAAgAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAACAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_uint8_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAAAwAAAAAAAAAAAAAAIAAAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAgAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAADAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAgAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAADAAAAAAAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAgAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAACChA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_int8_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAACQAAAAAAAAAAAAAAIAAAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAAAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAAACBA=="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_int8_per_axis_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAACUAQAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAAAEAQAACAEAAAwBAAADAAAAHAAAAEwAAADIAAAADgAYAAQACAAMAAAAAAAAABAAAAAUAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAACAAAAAgAAAA4AGAAEAAgADAAAABQAAAAQAAAAFAAAAAkAAAABAAAAAAAAACAAAAACAAAAAgAAAAIAAAAQABwAAAAAAAwAEAAAABgAEAAAAAAAAAAAAAAAEAAAABgAAAAAAAAAAAAAAAIAAAAAAAA/AACAPgIAAAAAAAAAAAAAAAAAAAAAAAAADgAYAAQACAAMAAAAAAAAABAAAAAUAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAACAAAAAgAAAAEAAAAAAAAAAQAAAAIAAAABAAAAEAAAAAoAEAAEAAgADAAAAAwAAAAAAAAACAAAABAAAAACAAAAAAAAAAEAAAABAAAAAgAAAAIAAAAQAAAAHAAAAAYACAAAAAAACAAAAAAAAAAGAAgABAAAAAgAAAAEAAAABAAAAAIEBAg="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_int8_per_axis_runtime_add_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAAQAgAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAACAAQAAhAEAAIgBAAADAAAAHAAAAIwAAAAAAQAADgAYAAQACAAMAAAAFAAAABAAAAAUAAAACQAAAAAAAAAAAAAAzAEAAAIAAAABAAAAAgAAABAAHAAAAAAADAAQAAAAAAAQAAAAAAAAAAAAAAAQAAAAFAAAAAAAAAAAAAAAAQAAAAAAAD8BAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAQAAAAAAAAAgAAAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAAOABgABAAIAAwAAAAUAAAAEAAAABQAAAAJAAAAAAAAAAAAAAAkAQAAAgAAAAEAAAACAAAAEAAcAAAAAAAMABAAAAAAABAAAAAAAAAAAAAAABAAAAAYAAAAAAAAAAAAAAABAAAAAAAAPwAAAAABAAAAAAAAAAAAAAABAAAAAAAAAAEAAAACAAAAAQAAABAAAAAKABAABAAIAAwAAAAMAAAAAAAAAAgAAAAQAAAAAgAAAAAAAAABAAAAAQAAAAIAAAACAAAAEAAAABwAAAAGAAgAAAAAAAgAAAAAAAAABgAIAAQAAAAIAAAABAAAAAIAAAACBAAAEAAQAAAAAAAEAAgAAAAMABAAAAAMAAAAGAAAAAEAAAACAAAAAAAAPwAAgD4AAAAAAgAAAAEAAAAAAAAA/v////////8QABAAAAAAAAQACAAAAAwAEAAAAAwAAAAUAAAAAQAAAAIAAAAAAAA/AACAPgIAAAABAAAAAAAAAP7/////////"),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_relu_n1_to_1_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFAAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAABAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAAEAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
extern "js" fn tflite_relu6_fixture() -> Bytes =
  #| () => Uint8Array.from(
  #|   atob("GAAAAFRGTDMOABgABAAIAAwAAAAUAAAAEAAAAAMAAAAQAAAANAAAAAAAAAD8AAAAAQAAABAAAAAMABQAAAAAAAAAEAAMAAAAAAAAAAAAAAAAAAAAFQAAAAEAAAAQAAAADAAUAAQACAAMABAADAAAABAAAABwAAAAdAAAAHgAAAACAAAAFAAAADwAAAAMABQABAAIAAwAAAAMAAAAEAAAAAAAAAAAAAAAAAAAAAIAAAABAAAABAAAAAwAFAAEAAgADAAAAAwAAAAQAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAAEAAAAAQAAAAAAAAABAAAAAQAAAAEAAAAQAAAACgAQAAQACAAMAAAADAAAAAAAAAAIAAAADAAAAAEAAAAAAAAAAQAAAAEAAAABAAAADAAAAAYACAAAAAAACAAAAAAAAAA="),
  #|   (character) => character.charCodeAt(0),
  #| )

///|
test "TFLite parser maps a float32 ADD FlatBuffer to LiteRT IR" {
  let model = TfliteModel::parse(tflite_add_fixture())
  let inputs = model.inputs()
  @test.assert_eq(inputs[0].name(), "input")
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].name(), "output")
  @test.assert_eq(outputs[0].tensor().values(), [8.0, 11.0])
}

///|
test "rank-one per-channel quantization normalizes a producer axis" {
  let shape = @shape.Shape::new([8])
  @test.assert_eq(
    normalize_quantized_dimension(
      shape,
      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
      3,
    ),
    0,
  )
}

///|
extern "js" fn tflite_concatenation_options_fixture(
  axis : Int,
  activation : Int,
) -> Bytes =
  #| (axis, activation) => {
  #|   const bytes = new Uint8Array(20);
  #|   const view = new DataView(bytes.buffer);
  #|   view.setUint16(0, 8, true);
  #|   view.setUint16(2, 12, true);
  #|   view.setUint16(4, 4, true);
  #|   view.setUint16(6, 8, true);
  #|   view.setInt32(8, 8, true);
  #|   view.setInt32(12, axis, true);
  #|   bytes[16] = activation;
  #|   return bytes;
  #| }

///|
test "TFLite CONCATENATION lowers its axis and fused activation" {
  let lhs_shape = @shape.Shape::new([2, 2])
  let rhs_shape = @shape.Shape::new([2, 1])
  let output_shape = @shape.Shape::new([2, 3])
  let tensors : Array[TfliteTensor] = [
    {
      name: "lhs",
      shape: lhs_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "rhs",
      shape: rhs_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "output",
      shape: output_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
  ]
  let options = FlatBufferReader::new(
    tflite_concatenation_options_fixture(1, 3),
  ).table(8)
  let lowered = concatenate_operator([0, 1], tensors[2], tensors, Some(options))
  @test.assert_eq(lowered.values.map(fn(value) { value.name() }), [
    "output_fused",
  ])
  match lowered.nodes {
    [
      LiteRtNode::Concat("lhs", "rhs", "output_fused", 1),
      LiteRtNode::Clamp("output_fused", "output", 0.0, 6.0),
    ] => ()
    _ => fail("expected CONCATENATION followed by RELU6 clamp")
  }
}

///|
test "TFLite CONCATENATION chains three inputs before lowering" {
  let lhs_shape = @shape.Shape::new([1, 1])
  let rhs_shape = @shape.Shape::new([1, 2])
  let output_shape = @shape.Shape::new([1, 4])
  let tensors : Array[TfliteTensor] = [
    {
      name: "lhs",
      shape: lhs_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "middle",
      shape: rhs_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "rhs",
      shape: lhs_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
    {
      name: "output",
      shape: output_shape,
      tensor_type: 0,
      buffer_index: 0,
      quantization: None,
    },
  ]
  let options = FlatBufferReader::new(
    tflite_concatenation_options_fixture(-1, 0),
  ).table(8)
  let lowered = concatenate_operator(
    [0, 1, 2],
    tensors[3],
    tensors,
    Some(options),
  )
  @test.assert_eq(lowered.values.map(fn(value) { value.name() }), [
    "output_concat_1",
  ])
  match lowered.nodes {
    [
      LiteRtNode::Concat("lhs", "middle", "output_concat_1", 1),
      LiteRtNode::Concat("output_concat_1", "rhs", "output", 1),
    ] => ()
    _ => fail("expected binary CONCATENATION chain")
  }
}

///|
test "TFLite parser maps a float32 SUB FlatBuffer to LiteRT IR" {
  let model = TfliteModel::parse(tflite_sub_fixture())
  let outputs = model.lower(
    (name, shape) => {
      if name == "tensor_0" {
        @cpu.CpuTensor::new(shape, [3.0, 20.0])
      } else {
        @cpu.CpuTensor::new(shape, [1.0, 10.0])
      }
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 10.0])
}

///|
test "TFLite parser keeps multiple inputs and outputs in declaration order" {
  let model = TfliteModel::parse(tflite_two_input_two_output_fixture())
  let inputs = model.inputs()
  @test.assert_eq(inputs.length(), 2)
  @test.assert_eq(inputs[0].name(), "tensor_0")
  @test.assert_eq(inputs[1].name(), "tensor_1")
  let outputs = model.lower(
    (name, shape) => {
      if name == "tensor_0" {
        @cpu.CpuTensor::new(shape, [1.0, 2.0])
      } else {
        @cpu.CpuTensor::new(shape, [10.0, 20.0])
      }
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs.length(), 2)
  @test.assert_eq(outputs[0].name(), "tensor_2")
  @test.assert_eq(outputs[0].tensor().values(), [11.0, 22.0])
  @test.assert_eq(outputs[1].name(), "tensor_3")
  @test.assert_eq(outputs[1].tensor().values(), [1.0, 2.0])
}

///|
test "TFLite parser lowers ADD fused RELU6 after addition" {
  let model = TfliteModel::parse(tflite_add_relu6_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [6.0, 6.0])
}

///|
test "TFLite parser lowers FULLY_CONNECTED weights into matmul and bias add" {
  let model = TfliteModel::parse(tflite_fully_connected_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [8.0, 11.0])
}

///|
test "TFLite parser lowers FULLY_CONNECTED fused RELU6 after bias add" {
  let model = TfliteModel::parse(tflite_fully_connected_relu6_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [6.0, 6.0])
}

///|
test "TFLite parser converts NHWC CONV_2D filters and bias" {
  let model = TfliteModel::parse(tflite_conv2d_fixture())
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [-4.0, -4.0, -4.0, -4.0])
}

///|
test "TFLite parser lowers RESHAPE with an int32 shape tensor" {
  let model = TfliteModel::parse(tflite_reshape_fixture()) catch {
    error => fail(error.to_string())
  }
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().shape().dimensions(), [4, 1])
  @test.assert_eq(outputs[0].tensor().values(), [1.0, 2.0, 3.0, 4.0])
}

///|
test "TFLite parser lowers TRANSPOSE with an int32 permutation tensor" {
  let model = TfliteModel::parse(tflite_transpose_fixture()) catch {
    error => fail(error.to_string())
  }
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().shape().dimensions(), [3, 2])
  @test.assert_eq(outputs[0].tensor().values(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

///|
test "TFLite parser lowers DEPTHWISE_CONV_2D as grouped NHWC/HWIO conv2d" {
  let model = TfliteModel::parse(tflite_depthwise_conv2d_fixture())
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(shape, [
        1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0,
        8.0, 80.0, 9.0, 90.0,
      ])
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [
    63.0, 730.0, 79.0, 930.0, 111.0, 1330.0, 127.0, 1530.0,
  ])
}

///|
test "TFLite parser lowers AVERAGE_POOL_2D with NHWC options" {
  let model = TfliteModel::parse(tflite_average_pool2d_fixture())
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [3.0, 4.0, 6.0, 7.0])
}

///|
test "TFLite parser lowers MAX_POOL_2D with NHWC options" {
  let model = TfliteModel::parse(tflite_max_pool2d_fixture())
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(shape, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [5.0, 6.0, 8.0, 9.0])
}

///|
test "TFLite parser lowers LOGISTIC" {
  let model = TfliteModel::parse(tflite_logistic_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [0.0, 1.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  let values = outputs[0].tensor().values()
  @test.assert_eq(values[0], 0.5)
  @test.assert_eq((values[1] - 0.7310586).abs() < 1.0e-6, true)
}

///|
test "TFLite parser lowers TANH" {
  let model = TfliteModel::parse(tflite_tanh_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [0.0, 1.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  let values = outputs[0].tensor().values()
  @test.assert_eq(values[0], 0.0)
  @test.assert_eq((values[1] - 0.7615942).abs() < 1.0e-6, true)
}

///|
test "TFLite parser dequantizes UINT8 tensors into float32 LiteRT values" {
  let model = TfliteModel::parse(tflite_uint8_add_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 4.0])
}

///|
test "TFLite parser dequantizes signed INT8 tensors into float32 LiteRT values" {
  let model = TfliteModel::parse(tflite_int8_add_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 2.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 4.0])
}

///|
test "TFLite parser dequantizes per-axis INT8 constants" {
  let model = TfliteModel::parse(tflite_int8_per_axis_add_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 1.0, 1.0, 1.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [2.0, 3.0, 2.0, 3.0])
}

///|
test "TFLite parser lowers RELU_N1_TO_1 to clamp" {
  let model = TfliteModel::parse(tflite_relu_n1_to_1_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [-2.0, -0.5, 0.5, 3.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [-1.0, -0.5, 0.5, 1.0])
}

///|
test "TFLite parser lowers RELU6 to clamp" {
  let model = TfliteModel::parse(tflite_relu6_fixture())
  let outputs = model.lower(
    (_name, shape) => @cpu.CpuTensor::new(shape, [-2.0, 0.5, 5.0, 8.0]),
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  @test.assert_eq(outputs[0].tensor().values(), [0.0, 0.5, 5.0, 6.0])
}

///|
test "TFLite UINT8 quantization helpers round-trip and saturate" {
  let values = dequantize_uint8([130, 132], 0.5, 128)
  @test.assert_eq(values, [1.0, 2.0])
  @test.assert_eq(requantize_uint8([2.0, 4.0], 0.5, 128), [132, 136])
  @test.assert_eq(requantize_uint8([-1000.0, 1000.0], 0.5, 128), [0, 255])
}

///|
test "TFLite runtime parsing preserves UINT8 input and output quantization" {
  let runtime = TfliteModel::parse_runtime(tflite_uint8_add_fixture())
  let inputs = runtime.inputs()
  let outputs = runtime.outputs()
  @test.assert_eq(inputs.length(), 1)
  @test.assert_eq(outputs.length(), 1)
  @test.assert_eq(inputs[0].dequantize([130, 132]), [1.0, 2.0])
  @test.assert_eq(outputs[0].requantize([2.0, 4.0]), [132, 136])
}

///|
test "TFLite runtime parsing preserves INT8 input and output quantization" {
  let runtime = TfliteModel::parse_runtime(tflite_int8_add_fixture())
  let inputs = runtime.inputs()
  let outputs = runtime.outputs()
  @test.assert_eq(inputs[0].dequantize([2, 4]), [1.0, 2.0])
  @test.assert_eq(outputs[0].requantize([2.0, 4.0]), [4, 8])
}

///|
test "TFLite runtime tensors convert per-axis INT8 values" {
  let runtime = runtime_tensor({
    name: "per_axis",
    shape: @shape.Shape::new([2, 2]),
    tensor_type: 9,
    buffer_index: 0,
    quantization: Some({
      scales: [0.5, 0.25],
      zero_points: [1, -2],
      quantized_dimension: 1,
    }),
  })
  @test.assert_eq(runtime.dequantize([3, 2, 3, 2]), [1.0, 1.0, 1.0, 1.0])
  @test.assert_eq(runtime.requantize([1.0, 1.0, 1.0, 1.0]), [3, 2, 3, 2])
}

///|
test "TFLite runtime parsing preserves per-axis INT8 input and output metadata" {
  let runtime = TfliteModel::parse_runtime(
    tflite_int8_per_axis_runtime_add_fixture(),
  ) catch {
    error => fail(error.to_string())
  }
  let inputs = runtime.inputs()
  let outputs = runtime.outputs()
  @test.assert_eq(inputs[0].dequantize([3, 2]), [1.0, 1.0])
  @test.assert_eq(outputs[0].requantize([2.0, 3.0]), [5, 10])
  let lowered = runtime
    .model()
    .lower((_name, shape) => @cpu.CpuTensor::new(shape, [1.0, 1.0]), (
      _name,
      shape,
      values,
    ) => @cpu.CpuTensor::new(shape, values))
  @test.assert_eq(lowered[0].tensor().values(), [2.0, 3.0])
}

///|
test "TFLite parser accepts TensorFlow's official ADD test model" {
  let model = TfliteModel::parse(tensorflow_add_fixture())
  @test.assert_eq(model.inputs().length(), 1)
  let outputs = model.lower(
    (_name, shape) => {
      @cpu.CpuTensor::new(
        shape,
        Array::make(shape.element_count(), Float::from_int(1)),
      )
    },
    (_name, shape, values) => @cpu.CpuTensor::new(shape, values),
  )
  let values = outputs[0].tensor().values()
  @test.assert_eq(values.length(), 192)
  @test.assert_eq(values[0], 3.0)
  @test.assert_eq(values[191], 3.0)
}
