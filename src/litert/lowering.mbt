///|
pub suberror LiteRtError {
  LiteRtError(String)
}

///|
pub fn LiteRtError::to_string(self : LiteRtError) -> String {
  let LiteRtError(message) = self
  message
}

///|
pub(all) enum LiteRtValueSource {
  Input
  Constant(Array[Float])
  Intermediate
}

///|
pub(all) struct LiteRtValue {
  name_ : String
  shape_ : @shape.Shape
  source_ : LiteRtValueSource
}

///|
fn validate_name(name : String) -> Unit raise LiteRtError {
  if name == "" {
    raise LiteRtError("LiteRT value name must not be empty")
  }
}

///|
pub fn LiteRtValue::input(
  name : String,
  shape : @shape.Shape,
) -> LiteRtValue raise LiteRtError {
  validate_name(name)
  { name_: name, shape_: shape, source_: Input }
}

///|
pub fn LiteRtValue::constant(
  name : String,
  shape : @shape.Shape,
  values : Array[Float],
) -> LiteRtValue raise LiteRtError {
  validate_name(name)
  if shape.element_count() != values.length() {
    raise LiteRtError(
      "LiteRT constant \{name} has \{values.length()} values for shape \{shape.to_string()}",
    )
  }
  { name_: name, shape_: shape, source_: Constant(values.copy()) }
}

///|
pub fn LiteRtValue::intermediate(
  name : String,
  shape : @shape.Shape,
) -> LiteRtValue raise LiteRtError {
  validate_name(name)
  { name_: name, shape_: shape, source_: Intermediate }
}

///|
pub fn LiteRtValue::name(self : LiteRtValue) -> String {
  self.name_
}

///|
pub fn LiteRtValue::shape(self : LiteRtValue) -> @shape.Shape {
  self.shape_
}

///|
pub(all) enum LiteRtNode {
  Add(String, String, String)
  Sub(String, String, String)
  Mul(String, String, String)
  Div(String, String, String)
  ReduceMean(String, String, Array[Int], Bool)
  Gather(String, String, Array[Int], @shape.Shape, Int)
  Slice(String, String, Array[Int], Array[Int])
  Gelu(String, String)
  LayerNormalization(String, String, String, String, Array[Int], Float)
  Concat(String, String, String, Int)
  Matmul(String, String, String)
  Conv2d(String, String, String, @shape.Conv2dOptions)
  MaxPool2d(String, String, @shape.Pool2dOptions)
  AveragePool2d(String, String, @shape.Pool2dOptions)
  Sigmoid(String, String)
  Tanh(String, String)
  Clamp(String, String, Float, Float)
  Relu(String, String)
  Softmax(String, String, Int)
  Reshape(String, String)
  Transpose(String, String, Array[Int])
}

///|
fn input_names(node : LiteRtNode) -> Array[String] {
  match node {
    Add(lhs, rhs, _) => [lhs, rhs]
    Sub(lhs, rhs, _) => [lhs, rhs]
    Mul(lhs, rhs, _) => [lhs, rhs]
    Div(lhs, rhs, _) => [lhs, rhs]
    ReduceMean(input, _, _, _) => [input]
    Gather(input, _, _, _, _) => [input]
    Slice(input, _, _, _) => [input]
    Gelu(input, _) => [input]
    LayerNormalization(input, scale, bias, _, _, _) => [input, scale, bias]
    Concat(lhs, rhs, _, _) => [lhs, rhs]
    Matmul(lhs, rhs, _) => [lhs, rhs]
    Conv2d(input, filter, _, _) => [input, filter]
    MaxPool2d(input, _, _) => [input]
    AveragePool2d(input, _, _) => [input]
    Sigmoid(input, _) => [input]
    Tanh(input, _) => [input]
    Clamp(input, _, _, _) => [input]
    Relu(input, _) => [input]
    Softmax(input, _, _) => [input]
    Reshape(input, _) => [input]
    Transpose(input, _, _) => [input]
  }
}

///|
fn output_name(node : LiteRtNode) -> String {
  match node {
    Add(_, _, output) => output
    Sub(_, _, output) => output
    Mul(_, _, output) => output
    Div(_, _, output) => output
    ReduceMean(_, output, _, _) => output
    Gather(_, output, _, _, _) => output
    Slice(_, output, _, _) => output
    Gelu(_, output) => output
    LayerNormalization(_, _, _, output, _, _) => output
    Concat(_, _, output, _) => output
    Matmul(_, _, output) => output
    Conv2d(_, _, output, _) => output
    MaxPool2d(_, output, _) => output
    AveragePool2d(_, output, _) => output
    Sigmoid(_, output) => output
    Tanh(_, output) => output
    Clamp(_, output, _, _) => output
    Relu(_, output) => output
    Softmax(_, output, _) => output
    Reshape(_, output) => output
    Transpose(_, output, _) => output
  }
}

///|
pub(all) struct LiteRtModel {
  values_ : Array[LiteRtValue]
  nodes_ : Array[LiteRtNode]
  output_names_ : Array[String]
}

///|
pub(all) struct LiteRtLoweredValue[T] {
  name_ : String
  tensor_ : T
}

///|
pub fn[T] LiteRtLoweredValue::name(self : LiteRtLoweredValue[T]) -> String {
  self.name_
}

///|
pub fn[T] LiteRtLoweredValue::tensor(self : LiteRtLoweredValue[T]) -> T {
  self.tensor_
}

///|
fn value_for(
  values : Map[String, LiteRtValue],
  name : String,
) -> LiteRtValue raise LiteRtError {
  guard values.contains(name) else {
    raise LiteRtError("LiteRT graph refers to unknown value: \{name}")
  }
  values[name]
}

///|
fn shape_or_litert_error(
  operation : () -> @shape.Shape raise @shape.ShapeError,
) -> @shape.Shape raise LiteRtError {
  operation() catch {
    error => raise LiteRtError(error.to_string())
  }
}

///|
fn infer_shape(
  node : LiteRtNode,
  values : Map[String, LiteRtValue],
) -> @shape.Shape raise LiteRtError {
  match node {
    Add(lhs, rhs, _) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => @shape.Shape::broadcast(lhs_shape, rhs_shape))
    }
    Sub(lhs, rhs, _) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => @shape.Shape::broadcast(lhs_shape, rhs_shape))
    }
    Mul(lhs, rhs, _) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => @shape.Shape::broadcast(lhs_shape, rhs_shape))
    }
    Div(lhs, rhs, _) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => @shape.Shape::broadcast(lhs_shape, rhs_shape))
    }
    ReduceMean(input, _, axes, keep_dimensions) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => input_shape.reduce(axes, keep_dimensions))
    }
    Gather(input, _, _, indices_shape, axis) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => input_shape.gather(indices_shape, axis))
    }
    Slice(input, _, starts, sizes) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => input_shape.slice(starts, sizes))
    }
    Gelu(input, _) => value_for(values, input).shape_
    LayerNormalization(input, scale, bias, _, axes, epsilon) => {
      let input_shape = value_for(values, input).shape_
      let scale_shape = value_for(values, scale).shape_
      let bias_shape = value_for(values, bias).shape_
      shape_or_litert_error(() => {
        @shape.Shape::layer_normalization(
          input_shape, scale_shape, bias_shape, axes, epsilon,
        )
      })
    }
    Concat(lhs, rhs, _, axis) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => {
        @shape.Shape::concat(lhs_shape, rhs_shape, axis)
      })
    }
    Matmul(lhs, rhs, _) => {
      let lhs_shape = value_for(values, lhs).shape_
      let rhs_shape = value_for(values, rhs).shape_
      shape_or_litert_error(() => @shape.Shape::matmul(lhs_shape, rhs_shape))
    }
    Conv2d(input, filter, _, options) => {
      let input_shape = value_for(values, input).shape_
      let filter_shape = value_for(values, filter).shape_
      shape_or_litert_error(() => {
        @shape.Shape::conv2d(input_shape, filter_shape, options)
      })
    }
    MaxPool2d(input, _, options) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => @shape.Shape::pool2d(input_shape, options))
    }
    AveragePool2d(input, _, options) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => @shape.Shape::pool2d(input_shape, options))
    }
    Sigmoid(input, _) => value_for(values, input).shape_
    Tanh(input, _) => value_for(values, input).shape_
    Clamp(input, _, minimum, maximum) => {
      if minimum > maximum {
        raise LiteRtError("LiteRT clamp minimum must not exceed maximum")
      }
      value_for(values, input).shape_
    }
    Relu(input, _) => value_for(values, input).shape_
    Softmax(input, _, axis) => {
      let shape = value_for(values, input).shape_
      shape.validate_axis(axis) catch {
        error => raise LiteRtError(error.to_string())
      }
      shape
    }
    Reshape(input, output) => {
      let input_shape = value_for(values, input).shape_
      let output_shape = value_for(values, output).shape_
      if input_shape.element_count() != output_shape.element_count() {
        raise LiteRtError(
          "LiteRT reshape from \{input} to \{output} changes element count",
        )
      }
      output_shape
    }
    Transpose(input, _, permutation) => {
      let input_shape = value_for(values, input).shape_
      shape_or_litert_error(() => input_shape.transpose(permutation))
    }
  }
}

///|
fn validate_node(
  node : LiteRtNode,
  values : Map[String, LiteRtValue],
  available : Map[String, Bool],
) -> Unit raise LiteRtError {
  for input in input_names(node) {
    let _ = value_for(values, input)
    if !available.contains(input) {
      raise LiteRtError("LiteRT node uses \{input} before it is available")
    }
  }
  let output = output_name(node)
  let output_value = value_for(values, output)
  match output_value.source_ {
    Intermediate => ()
    _ => raise LiteRtError("LiteRT node output \{output} must be intermediate")
  }
  if available.contains(output) {
    raise LiteRtError("LiteRT node overwrites value: \{output}")
  }
  let inferred = infer_shape(node, values)
  if !inferred.same_as(output_value.shape_) {
    raise LiteRtError(
      "LiteRT node output \{output} has shape \{output_value.shape_.to_string()}, expected \{inferred.to_string()}",
    )
  }
  available[output] = true
}

///|
pub fn LiteRtModel::new(
  values : Array[LiteRtValue],
  nodes : Array[LiteRtNode],
  output_names : Array[String],
) -> LiteRtModel raise LiteRtError {
  if values.is_empty() {
    raise LiteRtError("LiteRT graph must declare at least one value")
  }
  if output_names.is_empty() {
    raise LiteRtError("LiteRT graph must declare at least one output")
  }
  let values_by_name : Map[String, LiteRtValue] = Map([])
  let available : Map[String, Bool] = Map([])
  for value in values {
    validate_name(value.name_)
    if values_by_name.contains(value.name_) {
      raise LiteRtError("duplicate LiteRT value name: \{value.name_}")
    }
    values_by_name[value.name_] = value
    match value.source_ {
      Input | Constant(_) => available[value.name_] = true
      Intermediate => ()
    }
  }
  for node in nodes {
    validate_node(node, values_by_name, available)
  }
  let seen_outputs : Map[String, Bool] = Map([])
  for output in output_names {
    validate_name(output)
    let _ = value_for(values_by_name, output)
    if !available.contains(output) {
      raise LiteRtError("LiteRT output is not produced: \{output}")
    }
    if seen_outputs.contains(output) {
      raise LiteRtError("duplicate LiteRT output name: \{output}")
    }
    seen_outputs[output] = true
  }
  {
    values_: values.copy(),
    nodes_: nodes.copy(),
    output_names_: output_names.copy(),
  }
}

///|
pub fn LiteRtModel::inputs(self : LiteRtModel) -> Array[LiteRtValue] {
  self.values_.filter(fn(value) {
    match value.source_ {
      Input => true
      _ => false
    }
  })
}

///|
fn[T] tensor_for(
  tensors : Map[String, T],
  name : String,
) -> T raise LiteRtError {
  guard tensors.contains(name) else {
    raise LiteRtError("LiteRT lowering did not materialize value: \{name}")
  }
  tensors[name]
}

///|
fn[T : @tensor.TensorOps] lower_node(
  node : LiteRtNode,
  tensors : Map[String, T],
  output_shape : @shape.Shape,
) -> T raise {
  match node {
    Add(lhs, rhs, _) => tensor_for(tensors, lhs).add(tensor_for(tensors, rhs))
    Sub(lhs, rhs, _) => tensor_for(tensors, lhs).sub(tensor_for(tensors, rhs))
    Mul(lhs, rhs, _) => tensor_for(tensors, lhs).mul(tensor_for(tensors, rhs))
    Div(lhs, rhs, _) => tensor_for(tensors, lhs).div(tensor_for(tensors, rhs))
    ReduceMean(input, _, axes, keep_dimensions) =>
      tensor_for(tensors, input).reduce_mean(axes, keep_dimensions)
    Gather(input, _, indices, indices_shape, axis) =>
      tensor_for(tensors, input).gather(indices, indices_shape, axis)
    Slice(input, _, starts, sizes) =>
      tensor_for(tensors, input).slice(starts, sizes)
    Gelu(input, _) => tensor_for(tensors, input).gelu()
    LayerNormalization(input, scale, bias, _, axes, epsilon) =>
      tensor_for(tensors, input).layer_normalization(
        tensor_for(tensors, scale),
        tensor_for(tensors, bias),
        axes,
        epsilon,
      )
    Concat(lhs, rhs, _, axis) =>
      tensor_for(tensors, lhs).concat(tensor_for(tensors, rhs), axis)
    Matmul(lhs, rhs, _) =>
      tensor_for(tensors, lhs).matmul(tensor_for(tensors, rhs))
    Conv2d(input, filter, _, options) =>
      tensor_for(tensors, input).conv2d(tensor_for(tensors, filter), options)
    MaxPool2d(input, _, options) =>
      tensor_for(tensors, input).max_pool2d(options)
    AveragePool2d(input, _, options) =>
      tensor_for(tensors, input).average_pool2d(options)
    Sigmoid(input, _) => tensor_for(tensors, input).sigmoid()
    Tanh(input, _) => tensor_for(tensors, input).tanh()
    Clamp(input, _, minimum, maximum) =>
      tensor_for(tensors, input).clamp(minimum, maximum)
    Relu(input, _) => tensor_for(tensors, input).relu()
    Softmax(input, _, axis) => tensor_for(tensors, input).softmax(axis)
    Reshape(input, _) => tensor_for(tensors, input).reshape(output_shape)
    Transpose(input, _, permutation) =>
      tensor_for(tensors, input).transpose(permutation)
  }
}

///|
pub fn[T : @tensor.TensorOps] LiteRtModel::lower(
  self : LiteRtModel,
  make_input : (String, @shape.Shape) -> T raise,
  make_constant : (String, @shape.Shape, Array[Float]) -> T raise,
) -> Array[LiteRtLoweredValue[T]] raise {
  let tensors : Map[String, T] = Map([])
  let values_by_name : Map[String, LiteRtValue] = Map([])
  for value in self.values_ {
    values_by_name[value.name_] = value
    match value.source_ {
      Input => {
        let tensor = make_input(value.name_, value.shape_)
        if !tensor.shape().same_as(value.shape_) {
          raise LiteRtError(
            "LiteRT input factory returned wrong shape for \{value.name_}",
          )
        }
        tensors[value.name_] = tensor
      }
      Constant(values) => {
        let tensor = make_constant(value.name_, value.shape_, values.copy())
        if !tensor.shape().same_as(value.shape_) {
          raise LiteRtError(
            "LiteRT constant factory returned wrong shape for \{value.name_}",
          )
        }
        tensors[value.name_] = tensor
      }
      Intermediate => ()
    }
  }
  for node in self.nodes_ {
    let output = output_name(node)
    let expected = value_for(values_by_name, output)
    let tensor = lower_node(node, tensors, expected.shape_)
    if !tensor.shape().same_as(expected.shape_) {
      raise LiteRtError("LiteRT lowering produced wrong shape for \{output}")
    }
    tensors[output] = tensor
  }
  self.output_names_.map(fn(name) raise {
    { name_: name, tensor_: tensor_for(tensors, name) }
  })
}
