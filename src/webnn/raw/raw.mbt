///|
#external
pub type ML

///|
#external
pub type MLContext

///|
#external
pub type MLGraphBuilder

///|
#external
pub type MLOperand

///|
#external
pub type MLGraph

///|
#external
pub type MLTensor

///|
pub fn ML::as_any(self : ML) -> @core.Any = "%identity"

///|
pub fn MLContext::as_any(self : MLContext) -> @core.Any = "%identity"

///|
pub fn MLGraphBuilder::as_any(self : MLGraphBuilder) -> @core.Any = "%identity"

///|
pub fn MLOperand::as_any(self : MLOperand) -> @core.Any = "%identity"

///|
pub fn MLGraph::as_any(self : MLGraph) -> @core.Any = "%identity"

///|
pub fn MLTensor::as_any(self : MLTensor) -> @core.Any = "%identity"

///|
extern "js" fn ffi_navigator_ml() -> @core.Any =
  #| () => navigator.ml

///|
pub fn ml() -> ML? {
  let value = ffi_navigator_ml()
  if @core.is_nullish(value) {
    None
  } else {
    Some(value.cast())
  }
}

///|
pub extern "js" fn uses_accelerated_contract() -> Bool =
  #| () => typeof MLContext !== "undefined" && "accelerated" in MLContext.prototype

///|
pub extern "js" fn context_options(
  accelerated_contract : Bool,
  device_type : String,
) -> @core.Any =
  #| (acceleratedContract, deviceType) => acceleratedContract
  #|   ? { accelerated: deviceType !== "cpu" }
  #|   : { deviceType }

///|
pub async fn ML::create_context(self : ML, options : @core.Any) -> MLContext {
  let promise : @js.Promise[MLContext] = self
    .as_any()
    ._call("createContext", [options])
    .cast()
  promise.wait()
}

///|
pub extern "js" fn MLGraphBuilder::new(context : MLContext) -> MLGraphBuilder =
  #| (context) => new MLGraphBuilder(context)

///|
pub extern "js" fn MLContext::preferred_input_layout(
  self : MLContext,
) -> String =
  #| (context) =>
  #|   typeof context.opSupportLimits === "function"
  #|     ? (context.opSupportLimits().preferredInputLayout ?? "nchw")
  #|     : "nchw"

///|
pub extern "js" fn MLContext::has_op_support_limits(self : MLContext) -> Bool =
  #| (context) => typeof context.opSupportLimits === "function"

///|
pub extern "js" fn MLContext::has_tensor_io(self : MLContext) -> Bool =
  #| (context) =>
  #|   typeof context.createTensor === "function" &&
  #|   typeof context.writeTensor === "function" &&
  #|   typeof context.readTensor === "function" &&
  #|   typeof context.dispatch === "function"

///|
extern "js" fn ffi_descriptor(
  dimensions : Array[Int],
  writable : Bool,
  readable : Bool,
) -> @core.Any =
  #| (shape, writable, readable) => ({
  #|   dataType: "float32",
  #|   shape,
  #|   dimensions: shape,
  #|   ...(writable ? { writable: true } : {}),
  #|   ...(readable ? { readable: true } : {}),
  #| })

///|
pub extern "js" fn float32_array(values : Array[Float]) -> @core.Any =
  #| (values) => new Float32Array(values)

///|
extern "js" fn int32_descriptor(dimensions : Array[Int]) -> @core.Any =
  #| (shape) => ({ dataType: "int32", shape, dimensions: shape })

///|
extern "js" fn int32_array(values : Array[Int]) -> @core.Any =
  #| (values) => new Int32Array(values)

///|
pub extern "js" fn float32_values(buffer : @core.Any) -> Array[Float] =
  #| (buffer) => Array.from(new Float32Array(buffer))

///|
pub fn MLGraphBuilder::input(
  self : MLGraphBuilder,
  name : String,
  dimensions : Array[Int],
) -> MLOperand {
  self
  .as_any()
  ._call("input", [@core.any(name), ffi_descriptor(dimensions, false, false)])
  .cast()
}

///|
pub fn MLGraphBuilder::constant_float32(
  self : MLGraphBuilder,
  dimensions : Array[Int],
  values : Array[Float],
) -> MLOperand {
  self
  .as_any()
  ._call("constant", [
    ffi_descriptor(dimensions, false, false),
    float32_array(values),
  ])
  .cast()
}

///|
pub fn MLGraphBuilder::constant_int32(
  self : MLGraphBuilder,
  dimensions : Array[Int],
  values : Array[Int],
) -> MLOperand {
  self
  .as_any()
  ._call("constant", [int32_descriptor(dimensions), int32_array(values)])
  .cast()
}

///|
fn binary_operand(
  builder : MLGraphBuilder,
  operation : String,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  builder.as_any()._call(operation, [lhs.as_any(), rhs.as_any()]).cast()
}

///|
extern "js" fn ffi_gather_options(axis : Int) -> @core.Any =
  #| (axis) => ({ axis })

///|
pub fn MLGraphBuilder::gather(
  self : MLGraphBuilder,
  input : MLOperand,
  indices : MLOperand,
  axis : Int,
) -> MLOperand {
  self
  .as_any()
  ._call("gather", [input.as_any(), indices.as_any(), ffi_gather_options(axis)])
  .cast()
}

///|
pub fn MLGraphBuilder::slice(
  self : MLGraphBuilder,
  input : MLOperand,
  starts : Array[Int],
  sizes : Array[Int],
) -> MLOperand {
  self
  .as_any()
  ._call("slice", [input.as_any(), @core.any(starts), @core.any(sizes)])
  .cast()
}

///|
pub fn MLGraphBuilder::gelu(
  self : MLGraphBuilder,
  input : MLOperand,
) -> MLOperand {
  self.as_any()._call("gelu", [input.as_any()]).cast()
}

///|
extern "js" fn ffi_layer_normalization_options(
  scale : @core.Any,
  bias : @core.Any,
  axes : Array[Int],
  epsilon : Float,
) -> @core.Any =
  #| (scale, bias, axes, epsilon) => ({ scale, bias, axes, epsilon })

///|
pub fn MLGraphBuilder::layer_normalization(
  self : MLGraphBuilder,
  input : MLOperand,
  scale : MLOperand,
  bias : MLOperand,
  axes : Array[Int],
  epsilon : Float,
) -> MLOperand {
  self
  .as_any()
  ._call("layerNormalization", [
    input.as_any(),
    ffi_layer_normalization_options(
      scale.as_any(),
      bias.as_any(),
      axes,
      epsilon,
    ),
  ])
  .cast()
}

///|
pub fn MLGraphBuilder::add(
  self : MLGraphBuilder,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  binary_operand(self, "add", lhs, rhs)
}

///|
pub fn MLGraphBuilder::sub(
  self : MLGraphBuilder,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  binary_operand(self, "sub", lhs, rhs)
}

///|
pub fn MLGraphBuilder::mul(
  self : MLGraphBuilder,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  binary_operand(self, "mul", lhs, rhs)
}

///|
pub fn MLGraphBuilder::div(
  self : MLGraphBuilder,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  binary_operand(self, "div", lhs, rhs)
}

///|
pub fn MLGraphBuilder::concat(
  self : MLGraphBuilder,
  inputs : Array[MLOperand],
  axis : Int,
) -> MLOperand {
  let operands = inputs.map(fn(input) { input.as_any() })
  self.as_any()._call("concat", [@core.any(operands), @core.any(axis)]).cast()
}

///|
pub fn MLGraphBuilder::matmul(
  self : MLGraphBuilder,
  lhs : MLOperand,
  rhs : MLOperand,
) -> MLOperand {
  binary_operand(self, "matmul", lhs, rhs)
}

///|
extern "js" fn ffi_conv2d_options(
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  groups : Int,
  input_layout : String,
  filter_layout : String,
) -> @core.Any =
  #| (padding, strides, dilations, groups, inputLayout, filterLayout) => ({
  #|   padding,
  #|   strides,
  #|   dilations,
  #|   groups,
  #|   inputLayout,
  #|   filterLayout,
  #| })

///|
pub fn MLGraphBuilder::conv2d(
  self : MLGraphBuilder,
  input : MLOperand,
  filter : MLOperand,
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  groups : Int,
  input_layout : String,
  filter_layout : String,
) -> MLOperand {
  self
  .as_any()
  ._call("conv2d", [
    input.as_any(),
    filter.as_any(),
    ffi_conv2d_options(
      padding, strides, dilations, groups, input_layout, filter_layout,
    ),
  ])
  .cast()
}

///|
extern "js" fn ffi_pool2d_options(
  window_dimensions : Array[Int],
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  layout : String,
) -> @core.Any =
  #| (windowDimensions, padding, strides, dilations, layout) => ({
  #|   windowDimensions,
  #|   padding,
  #|   strides,
  #|   dilations,
  #|   layout,
  #| })

///|
pub fn MLGraphBuilder::max_pool2d(
  self : MLGraphBuilder,
  input : MLOperand,
  window_dimensions : Array[Int],
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  layout : String,
) -> MLOperand {
  self
  .as_any()
  ._call("maxPool2d", [
    input.as_any(),
    ffi_pool2d_options(window_dimensions, padding, strides, dilations, layout),
  ])
  .cast()
}

///|
pub fn MLGraphBuilder::average_pool2d(
  self : MLGraphBuilder,
  input : MLOperand,
  window_dimensions : Array[Int],
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  layout : String,
) -> MLOperand {
  self
  .as_any()
  ._call("averagePool2d", [
    input.as_any(),
    ffi_pool2d_options(window_dimensions, padding, strides, dilations, layout),
  ])
  .cast()
}

///|
pub fn MLGraphBuilder::sigmoid(
  self : MLGraphBuilder,
  input : MLOperand,
) -> MLOperand {
  self.as_any()._call("sigmoid", [input.as_any()]).cast()
}

///|
pub fn MLGraphBuilder::tanh(
  self : MLGraphBuilder,
  input : MLOperand,
) -> MLOperand {
  self.as_any()._call("tanh", [input.as_any()]).cast()
}

///|
extern "js" fn ffi_clamp_options(minimum : Float, maximum : Float) -> @core.Any =
  #| (minValue, maxValue) => ({ minValue, maxValue })

///|
pub fn MLGraphBuilder::clamp(
  self : MLGraphBuilder,
  input : MLOperand,
  minimum : Float,
  maximum : Float,
) -> MLOperand {
  self
  .as_any()
  ._call("clamp", [input.as_any(), ffi_clamp_options(minimum, maximum)])
  .cast()
}

///|
pub fn MLGraphBuilder::relu(
  self : MLGraphBuilder,
  input : MLOperand,
) -> MLOperand {
  self.as_any()._call("relu", [input.as_any()]).cast()
}

///|
pub fn MLGraphBuilder::softmax(
  self : MLGraphBuilder,
  input : MLOperand,
  axis : Int,
) -> MLOperand {
  self.as_any()._call("softmax", [input.as_any(), @core.any(axis)]).cast()
}

///|
extern "js" fn ffi_reduce_options(
  axes : Array[Int],
  keep_dimensions : Bool,
) -> @core.Any =
  #| (axes, keepDimensions) => ({ axes, keepDimensions })

///|
pub fn MLGraphBuilder::reduce_mean(
  self : MLGraphBuilder,
  input : MLOperand,
  axes : Array[Int],
  keep_dimensions : Bool,
) -> MLOperand {
  self
  .as_any()
  ._call("reduceMean", [
    input.as_any(),
    ffi_reduce_options(axes, keep_dimensions),
  ])
  .cast()
}

///|
pub fn MLGraphBuilder::reshape(
  self : MLGraphBuilder,
  input : MLOperand,
  dimensions : Array[Int],
) -> MLOperand {
  self.as_any()._call("reshape", [input.as_any(), @core.any(dimensions)]).cast()
}

///|
extern "js" fn ffi_transpose_options(permutation : Array[Int]) -> @core.Any =
  #| (permutation) => ({ permutation })

///|
pub fn MLGraphBuilder::transpose(
  self : MLGraphBuilder,
  input : MLOperand,
  permutation : Array[Int],
) -> MLOperand {
  self
  .as_any()
  ._call("transpose", [input.as_any(), ffi_transpose_options(permutation)])
  .cast()
}

///|
pub extern "js" fn same_builder(
  lhs : MLGraphBuilder,
  rhs : MLGraphBuilder,
) -> Bool =
  #| (lhs, rhs) => lhs === rhs

///|
extern "js" fn ffi_named_values(
  names : Array[String],
  values : Array[@core.Any],
) -> @core.Any =
  #| (names, values) => Object.fromEntries(
  #|   names.map((name, index) => [name, values[index]]),
  #| )

///|
pub async fn MLGraphBuilder::build(
  self : MLGraphBuilder,
  output_name : String,
  output : MLOperand,
) -> MLGraph {
  self.build_many([output_name], [output])
}

///|
pub async fn MLGraphBuilder::build_many(
  self : MLGraphBuilder,
  output_names : Array[String],
  outputs : Array[MLOperand],
) -> MLGraph {
  let output_values = outputs.map(fn(output) { output.as_any() })
  let promise : @js.Promise[MLGraph] = self
    .as_any()
    ._call("build", [ffi_named_values(output_names, output_values)])
    .cast()
  promise.wait()
}

///|
pub async fn MLContext::create_tensor(
  self : MLContext,
  dimensions : Array[Int],
  writable : Bool,
  readable : Bool,
) -> MLTensor {
  let promise : @js.Promise[MLTensor] = self
    .as_any()
    ._call("createTensor", [ffi_descriptor(dimensions, writable, readable)])
    .cast()
  promise.wait()
}

///|
pub fn MLContext::write_tensor(
  self : MLContext,
  tensor : MLTensor,
  values : Array[Float],
) -> Unit {
  self.as_any()._call("writeTensor", [tensor.as_any(), float32_array(values)])
  |> ignore
}

///|
pub fn MLContext::dispatch_single(
  self : MLContext,
  graph : MLGraph,
  input_name : String,
  input : MLTensor,
  output_name : String,
  output : MLTensor,
) -> Unit {
  self.dispatch_many(graph, [input_name], [input], [output_name], [output])
}

///|
pub fn MLContext::dispatch_many(
  self : MLContext,
  graph : MLGraph,
  input_names : Array[String],
  inputs : Array[MLTensor],
  output_names : Array[String],
  outputs : Array[MLTensor],
) -> Unit {
  let input_values = inputs.map(fn(input) { input.as_any() })
  let output_values = outputs.map(fn(output) { output.as_any() })
  self
  .as_any()
  ._call("dispatch", [
    graph.as_any(),
    ffi_named_values(input_names, input_values),
    ffi_named_values(output_names, output_values),
  ])
  |> ignore
}

///|
pub async fn MLContext::read_tensor(
  self : MLContext,
  tensor : MLTensor,
) -> Array[Float] {
  let promise : @js.Promise[@core.Any] = self
    .as_any()
    ._call("readTensor", [tensor.as_any()])
    .cast()
  float32_values(promise.wait())
}

///|
pub extern "js" fn MLContext::supported_operators(
  self : MLContext,
  candidates : Array[String],
) -> Array[String] =
  #| (context, candidates) => {
  #|   if (typeof context.opSupportLimits !== "function") return [];
  #|   const limits = context.opSupportLimits();
  #|   return candidates.filter((name) => limits[name] !== undefined);
  #| }

///|
pub fn MLContext::destroy(self : MLContext) -> Unit {
  self.as_any()._call("destroy", []) |> ignore
}

///|
pub fn MLTensor::destroy(self : MLTensor) -> Unit {
  self.as_any()._call("destroy", []) |> ignore
}
