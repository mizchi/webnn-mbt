///|
pub struct WebNNGraphBuilder {
  context : @raw.MLContext
  builder : @raw.MLGraphBuilder
  supported_operators_ : Array[String]
}

///|
pub(all) enum WebNNContextContract {
  Accelerated
  LegacyDeviceType
} derive(Debug, Eq)

///|
pub struct WebNNCapabilities {
  contract_ : WebNNContextContract
  preference_ : @compat.DevicePreference
  preferred_input_layout_ : @shape.InputLayout
  op_support_limits_available_ : Bool
  tensor_io_available_ : Bool
  supported_operators_ : Array[String]
}

///|
pub struct WebNNTensor {
  builder : @raw.MLGraphBuilder
  operand : @raw.MLOperand
  shape_ : @shape.Shape
  supported_operators : Array[String]
}

///|
pub struct WebNNInput {
  name_ : String
  tensor_ : WebNNTensor
}

///|
pub struct WebNNOutput {
  name_ : String
  tensor_ : WebNNTensor
}

///|
pub struct WebNNNamedValues {
  name_ : String
  values_ : Array[Float]
}

///|
struct WebNNValueSpec {
  name : String
  shape : @shape.Shape
}

///|
pub struct WebNNSession {
  context : @raw.MLContext
  graph : @raw.MLGraph
  input_specs : Array[WebNNValueSpec]
  output_specs : Array[WebNNValueSpec]
}

///|
pub struct WebNNExecution {
  context : @raw.MLContext
  graph : @raw.MLGraph
  input_specs : Array[WebNNValueSpec]
  inputs : Array[@raw.MLTensor]
  output_specs : Array[WebNNValueSpec]
  outputs : Array[@raw.MLTensor]
}

///|
pub async fn WebNNGraphBuilder::new(
  preference : @compat.DevicePreference,
) -> WebNNGraphBuilder {
  let context = @compat.create_context(preference)
  let supported_operators = context.supported_operators(backend_operators())
  {
    context,
    builder: @raw.MLGraphBuilder::new(context),
    supported_operators_: supported_operators,
  }
}

///|
pub fn backend_operators() -> Array[String] {
  [
    "add", "sub", "mul", "div", "concat", "matmul", "conv2d", "maxPool2d", "averagePool2d",
    "sigmoid", "tanh", "gelu", "layerNormalization", "clamp", "relu", "softmax",
    "reduceMean", "gather", "slice", "reshape", "transpose",
  ]
}

///|
pub fn WebNNCapabilities::contract(
  self : WebNNCapabilities,
) -> WebNNContextContract {
  self.contract_
}

///|
pub fn WebNNContextContract::to_string(self : WebNNContextContract) -> String {
  match self {
    Accelerated => "accelerated"
    LegacyDeviceType => "legacy-device-type"
  }
}

///|
pub fn WebNNCapabilities::preference(
  self : WebNNCapabilities,
) -> @compat.DevicePreference {
  self.preference_
}

///|
pub fn WebNNCapabilities::preferred_input_layout(
  self : WebNNCapabilities,
) -> @shape.InputLayout {
  self.preferred_input_layout_
}

///|
pub fn WebNNCapabilities::op_support_limits_available(
  self : WebNNCapabilities,
) -> Bool {
  self.op_support_limits_available_
}

///|
pub fn WebNNCapabilities::tensor_io_available(self : WebNNCapabilities) -> Bool {
  self.tensor_io_available_
}

///|
pub fn WebNNCapabilities::supported_operators(
  self : WebNNCapabilities,
) -> Array[String] {
  self.supported_operators_.copy()
}

///|
/// Probe the browser context contract required by this backend.
///
/// The result reports only capabilities used by the library: the context
/// selection contract, opSupportLimits(), prepared tensor I/O, the preferred
/// input layout, and the operators returned for backend_operators().
pub async fn probe_capabilities(
  preference : @compat.DevicePreference,
) -> WebNNCapabilities {
  let context = @compat.create_context(preference)
  let preferred_input_layout = match context.preferred_input_layout() {
    "nhwc" => @shape.InputLayout::Nhwc
    _ => @shape.InputLayout::Nchw
  }
  let contract = if @raw.uses_accelerated_contract() {
    Accelerated
  } else {
    LegacyDeviceType
  }
  let capabilities = {
    contract_: contract,
    preference_: preference,
    preferred_input_layout_: preferred_input_layout,
    op_support_limits_available_: context.has_op_support_limits(),
    tensor_io_available_: context.has_tensor_io(),
    supported_operators_: context.supported_operators(backend_operators()),
  }
  context.destroy()
  capabilities
}

///|
pub fn WebNNGraphBuilder::supported_operators(
  self : WebNNGraphBuilder,
) -> Array[String] {
  self.supported_operators_.copy()
}

///|
pub fn WebNNGraphBuilder::preferred_input_layout(
  self : WebNNGraphBuilder,
) -> @shape.InputLayout {
  match self.context.preferred_input_layout() {
    "nhwc" => @shape.InputLayout::Nhwc
    _ => @shape.InputLayout::Nchw
  }
}

///|
pub fn WebNNGraphBuilder::destroy(self : WebNNGraphBuilder) -> Unit {
  self.context.destroy()
}

///|
pub fn WebNNGraphBuilder::input(
  self : WebNNGraphBuilder,
  name : String,
  shape : @shape.Shape,
) -> WebNNInput raise @tensor.TensorError {
  if name == "" {
    raise @tensor.TensorError::new("WebNN input name must not be empty")
  }
  let tensor = WebNNTensor::{
    builder: self.builder,
    operand: self.builder.input(name, shape.dimensions()),
    shape_: shape,
    supported_operators: self.supported_operators_,
  }
  { name_: name, tensor_: tensor }
}

///|
pub fn WebNNInput::tensor(self : WebNNInput) -> WebNNTensor {
  self.tensor_
}

///|
pub fn WebNNGraphBuilder::output(
  self : WebNNGraphBuilder,
  name : String,
  tensor : WebNNTensor,
) -> WebNNOutput raise @tensor.TensorError {
  if name == "" {
    raise @tensor.TensorError::new("WebNN output name must not be empty")
  }
  if !@raw.same_builder(self.builder, tensor.builder) {
    raise @tensor.TensorError::new(
      "output belongs to a different WebNN graph builder",
    )
  }
  { name_: name, tensor_: tensor }
}

///|
pub fn WebNNNamedValues::new(
  name : String,
  values : Array[Float],
) -> WebNNNamedValues raise @tensor.TensorError {
  if name == "" {
    raise @tensor.TensorError::new("WebNN runtime value name must not be empty")
  }
  { name_: name, values_: values.copy() }
}

///|
pub fn WebNNNamedValues::name(self : WebNNNamedValues) -> String {
  self.name_
}

///|
pub fn WebNNNamedValues::values(self : WebNNNamedValues) -> Array[Float] {
  self.values_.copy()
}

///|
pub fn WebNNGraphBuilder::constant(
  self : WebNNGraphBuilder,
  shape : @shape.Shape,
  values : Array[Float],
) -> WebNNTensor raise @tensor.TensorError {
  if shape.element_count() != values.length() {
    raise @tensor.TensorError::new(
      "constant data length \{values.length()} does not match shape \{shape.to_string()}",
    )
  }
  {
    builder: self.builder,
    operand: self.builder.constant_float32(shape.dimensions(), values),
    shape_: shape,
    supported_operators: self.supported_operators_,
  }
}

///|
pub async fn WebNNGraphBuilder::compile_single(
  self : WebNNGraphBuilder,
  input : WebNNInput,
  output_name : String,
  output : WebNNTensor,
) -> WebNNSession {
  self.compile_named([input], [self.output(output_name, output)])
}

///|
fn ensure_unique_binding_names(
  kind : String,
  names : Array[String],
) -> Unit raise @tensor.TensorError {
  let seen : Map[String, Bool] = Map([])
  for name in names {
    if name == "" {
      raise @tensor.TensorError::new("WebNN \{kind} name must not be empty")
    }
    if seen.contains(name) {
      raise @tensor.TensorError::new("duplicate WebNN \{kind} name: \{name}")
    }
    seen[name] = true
  }
}

///|
pub async fn WebNNGraphBuilder::compile_named(
  self : WebNNGraphBuilder,
  inputs : Array[WebNNInput],
  outputs : Array[WebNNOutput],
) -> WebNNSession {
  if inputs.is_empty() {
    raise @tensor.TensorError::new("WebNN graph must have at least one input")
  }
  if outputs.is_empty() {
    raise @tensor.TensorError::new("WebNN graph must have at least one output")
  }
  let input_names = inputs.map(fn(input) { input.name_ })
  let output_names = outputs.map(fn(output) { output.name_ })
  ensure_unique_binding_names("input", input_names)
  ensure_unique_binding_names("output", output_names)
  for input in inputs {
    if !@raw.same_builder(self.builder, input.tensor_.builder) {
      raise @tensor.TensorError::new(
        "input belongs to a different WebNN graph builder",
      )
    }
  }
  for output in outputs {
    if !@raw.same_builder(self.builder, output.tensor_.builder) {
      raise @tensor.TensorError::new(
        "output belongs to a different WebNN graph builder",
      )
    }
  }
  let graph = self.builder.build_many(
    output_names,
    outputs.map(fn(output) { output.tensor_.operand }),
  )
  {
    context: self.context,
    graph,
    input_specs: inputs.map(fn(input) {
      { name: input.name_, shape: input.tensor_.shape_ }
    }),
    output_specs: outputs.map(fn(output) {
      { name: output.name_, shape: output.tensor_.shape_ }
    }),
  }
}

///|
pub async fn WebNNSession::run(
  self : WebNNSession,
  input_values : Array[Float],
) -> Array[Float] {
  if self.input_specs.length() != 1 || self.output_specs.length() != 1 {
    raise @tensor.TensorError::new(
      "single-value run requires exactly one input and one output",
    )
  }
  let execution = self.prepare()
  defer execution.destroy()
  execution.run(input_values)
}

///|
pub async fn WebNNSession::run_named(
  self : WebNNSession,
  input_values : Array[WebNNNamedValues],
) -> Array[WebNNNamedValues] {
  let execution = self.prepare()
  defer execution.destroy()
  execution.run_named(input_values)
}

///|
pub async fn WebNNSession::prepare(self : WebNNSession) -> WebNNExecution {
  let inputs : Array[@raw.MLTensor] = []
  for spec in self.input_specs {
    let input = self.context.create_tensor(spec.shape.dimensions(), true, false) catch {
      error => {
        inputs.each(fn(prepared) { prepared.destroy() })
        raise error
      }
    }
    inputs.push(input)
  }
  let outputs : Array[@raw.MLTensor] = []
  for spec in self.output_specs {
    let output = self.context.create_tensor(
      spec.shape.dimensions(),
      false,
      true,
    ) catch {
      error => {
        inputs.each(fn(prepared) { prepared.destroy() })
        outputs.each(fn(prepared) { prepared.destroy() })
        raise error
      }
    }
    outputs.push(output)
  }
  {
    context: self.context,
    graph: self.graph,
    input_specs: self.input_specs,
    inputs,
    output_specs: self.output_specs,
    outputs,
  }
}

///|
pub async fn WebNNExecution::run(
  self : WebNNExecution,
  input_values : Array[Float],
) -> Array[Float] {
  if self.input_specs.length() != 1 || self.output_specs.length() != 1 {
    raise @tensor.TensorError::new(
      "single-value run requires exactly one input and one output",
    )
  }
  let results = self.run_named([
    { name_: self.input_specs[0].name, values_: input_values },
  ])
  results[0].values_
}

///|
pub async fn WebNNExecution::run_named(
  self : WebNNExecution,
  input_values : Array[WebNNNamedValues],
) -> Array[WebNNNamedValues] {
  if input_values.length() != self.input_specs.length() {
    raise @tensor.TensorError::new(
      "WebNN runtime input count \{input_values.length()} does not match graph input count \{self.input_specs.length()}",
    )
  }
  let expected : Map[String, Bool] = Map([])
  for spec in self.input_specs {
    expected[spec.name] = true
  }
  let values_by_name : Map[String, Array[Float]] = Map([])
  for binding in input_values {
    if !expected.contains(binding.name_) {
      raise @tensor.TensorError::new(
        "unknown WebNN runtime input name: \{binding.name_}",
      )
    }
    if values_by_name.contains(binding.name_) {
      raise @tensor.TensorError::new(
        "duplicate WebNN runtime input name: \{binding.name_}",
      )
    }
    values_by_name[binding.name_] = binding.values_
  }
  let input_names : Array[String] = []
  for index, spec in self.input_specs {
    guard values_by_name.get(spec.name) is Some(values) else {
      raise @tensor.TensorError::new(
        "missing WebNN runtime input name: \{spec.name}",
      )
    }
    if values.length() != spec.shape.element_count() {
      raise @tensor.TensorError::new(
        "input \{spec.name} data length \{values.length()} does not match shape \{spec.shape.to_string()}",
      )
    }
    self.context.write_tensor(self.inputs[index], values)
    input_names.push(spec.name)
  }
  let output_names = self.output_specs.map(fn(spec) { spec.name })
  self.context.dispatch_many(
    self.graph,
    input_names,
    self.inputs,
    output_names,
    self.outputs,
  )
  let results : Array[WebNNNamedValues] = []
  for index, spec in self.output_specs {
    let values = self.context.read_tensor(self.outputs[index])
    results.push({ name_: spec.name, values_: values })
  }
  results
}

///|
pub fn WebNNExecution::destroy(self : WebNNExecution) -> Unit {
  self.inputs.each(fn(input) { input.destroy() })
  self.outputs.each(fn(output) { output.destroy() })
}

///|
fn ensure_same_builder(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> Unit raise @tensor.TensorError {
  if !@raw.same_builder(lhs.builder, rhs.builder) {
    raise @tensor.TensorError::new(
      "operands belong to different WebNN graph builders",
    )
  }
}

///|
fn ensure_operator(
  tensor : WebNNTensor,
  operation : String,
) -> Unit raise @tensor.TensorError {
  if !tensor.supported_operators.contains(operation) {
    raise @tensor.TensorError::new(
      "WebNN context does not report support for the \{operation} operator",
    )
  }
}

///|
fn shape_or_tensor_error(
  operation : () -> @shape.Shape raise @shape.ShapeError,
) -> @shape.Shape raise @tensor.TensorError {
  operation() catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn shape(self : WebNNTensor) -> @shape.Shape {
  self.shape_
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn add(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "add")
  {
    builder: lhs.builder,
    operand: lhs.builder.add(lhs.operand, rhs.operand),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::broadcast(lhs.shape_, rhs.shape_)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn sub(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "sub")
  {
    builder: lhs.builder,
    operand: lhs.builder.sub(lhs.operand, rhs.operand),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::broadcast(lhs.shape_, rhs.shape_)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn mul(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "mul")
  {
    builder: lhs.builder,
    operand: lhs.builder.mul(lhs.operand, rhs.operand),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::broadcast(lhs.shape_, rhs.shape_)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn div(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "div")
  {
    builder: lhs.builder,
    operand: lhs.builder.div(lhs.operand, rhs.operand),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::broadcast(lhs.shape_, rhs.shape_)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn concat(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
  axis : Int,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "concat")
  {
    builder: lhs.builder,
    operand: lhs.builder.concat([lhs.operand, rhs.operand], axis),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::concat(lhs.shape_, rhs.shape_, axis)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn matmul(
  lhs : WebNNTensor,
  rhs : WebNNTensor,
) -> WebNNTensor {
  ensure_same_builder(lhs, rhs)
  ensure_operator(lhs, "matmul")
  {
    builder: lhs.builder,
    operand: lhs.builder.matmul(lhs.operand, rhs.operand),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::matmul(lhs.shape_, rhs.shape_)
    }),
    supported_operators: lhs.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn conv2d(
  input : WebNNTensor,
  filter : WebNNTensor,
  options : @shape.Conv2dOptions,
) -> WebNNTensor {
  ensure_same_builder(input, filter)
  ensure_operator(input, "conv2d")
  {
    builder: input.builder,
    operand: input.builder.conv2d(
      input.operand,
      filter.operand,
      options.padding(),
      options.strides(),
      options.dilations(),
      options.groups(),
      options.input_layout().to_webnn_string(),
      options.filter_layout().to_webnn_string(),
    ),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::conv2d(input.shape_, filter.shape_, options)
    }),
    supported_operators: input.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn max_pool2d(
  self : WebNNTensor,
  options : @shape.Pool2dOptions,
) -> WebNNTensor {
  ensure_operator(self, "maxPool2d")
  {
    builder: self.builder,
    operand: self.builder.max_pool2d(
      self.operand,
      options.window_dimensions(),
      options.padding(),
      options.strides(),
      options.dilations(),
      options.input_layout().to_webnn_string(),
    ),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::pool2d(self.shape_, options)
    }),
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn average_pool2d(
  self : WebNNTensor,
  options : @shape.Pool2dOptions,
) -> WebNNTensor {
  ensure_operator(self, "averagePool2d")
  {
    builder: self.builder,
    operand: self.builder.average_pool2d(
      self.operand,
      options.window_dimensions(),
      options.padding(),
      options.strides(),
      options.dilations(),
      options.input_layout().to_webnn_string(),
    ),
    shape_: shape_or_tensor_error(() => {
      @shape.Shape::pool2d(self.shape_, options)
    }),
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn sigmoid(self : WebNNTensor) -> WebNNTensor {
  ensure_operator(self, "sigmoid")
  {
    builder: self.builder,
    operand: self.builder.sigmoid(self.operand),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn tanh(self : WebNNTensor) -> WebNNTensor {
  ensure_operator(self, "tanh")
  {
    builder: self.builder,
    operand: self.builder.tanh(self.operand),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn clamp(
  self : WebNNTensor,
  minimum : Float,
  maximum : Float,
) -> WebNNTensor {
  if minimum > maximum {
    raise @tensor.TensorError::new("clamp minimum must not exceed maximum")
  }
  ensure_operator(self, "clamp")
  {
    builder: self.builder,
    operand: self.builder.clamp(self.operand, minimum, maximum),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn relu(self : WebNNTensor) -> WebNNTensor {
  ensure_operator(self, "relu")
  {
    builder: self.builder,
    operand: self.builder.relu(self.operand),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn softmax(
  self : WebNNTensor,
  axis : Int,
) -> WebNNTensor {
  ensure_operator(self, "softmax")
  self.shape_.validate_axis(axis) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  {
    builder: self.builder,
    operand: self.builder.softmax(self.operand, axis),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn reduce_mean(
  self : WebNNTensor,
  axes : Array[Int],
  keep_dimensions : Bool,
) -> WebNNTensor {
  ensure_operator(self, "reduceMean")
  {
    builder: self.builder,
    operand: self.builder.reduce_mean(self.operand, axes, keep_dimensions),
    shape_: shape_or_tensor_error(() => {
      self.shape_.reduce(axes, keep_dimensions)
    }),
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn gather(
  self : WebNNTensor,
  indices : Array[Int],
  indices_shape : @shape.Shape,
  axis : Int,
) -> WebNNTensor {
  ensure_operator(self, "gather")
  {
    builder: self.builder,
    operand: self.builder.gather(
      self.operand,
      self.builder.constant_int32(indices_shape.dimensions(), indices),
      axis,
    ),
    shape_: shape_or_tensor_error(() => self.shape_.gather(indices_shape, axis)),
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn slice(
  self : WebNNTensor,
  starts : Array[Int],
  sizes : Array[Int],
) -> WebNNTensor {
  ensure_operator(self, "slice")
  {
    builder: self.builder,
    operand: self.builder.slice(self.operand, starts, sizes),
    shape_: shape_or_tensor_error(() => self.shape_.slice(starts, sizes)),
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn gelu(self : WebNNTensor) -> WebNNTensor {
  ensure_operator(self, "gelu")
  {
    builder: self.builder,
    operand: self.builder.gelu(self.operand),
    shape_: self.shape_,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn layer_normalization(
  input : WebNNTensor,
  scale : WebNNTensor,
  bias : WebNNTensor,
  axes : Array[Int],
  epsilon : Float,
) -> WebNNTensor {
  ensure_same_builder(input, scale)
  ensure_same_builder(input, bias)
  ensure_operator(input, "layerNormalization")
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::layer_normalization(
      input.shape_,
      scale.shape_,
      bias.shape_,
      axes,
      epsilon,
    )
  })
  {
    builder: input.builder,
    operand: input.builder.layer_normalization(
      input.operand,
      scale.operand,
      bias.operand,
      axes,
      epsilon,
    ),
    shape_: output_shape,
    supported_operators: input.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn reshape(
  self : WebNNTensor,
  shape : @shape.Shape,
) -> WebNNTensor {
  ensure_operator(self, "reshape")
  if self.shape_.element_count() != shape.element_count() {
    raise @tensor.TensorError::new(
      "reshape changes element count from \{self.shape_.element_count()} to \{shape.element_count()}",
    )
  }
  {
    builder: self.builder,
    operand: self.builder.reshape(self.operand, shape.dimensions()),
    shape_: shape,
    supported_operators: self.supported_operators,
  }
}

///|
pub impl @tensor.TensorOps for WebNNTensor with fn transpose(
  self : WebNNTensor,
  permutation : Array[Int],
) -> WebNNTensor {
  ensure_operator(self, "transpose")
  {
    builder: self.builder,
    operand: self.builder.transpose(self.operand, permutation),
    shape_: shape_or_tensor_error(() => self.shape_.transpose(permutation)),
    supported_operators: self.supported_operators,
  }
}
