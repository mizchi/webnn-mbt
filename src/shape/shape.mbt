///|
pub suberror ShapeError {
  ShapeError(String)
}

///|
pub fn ShapeError::to_string(self : ShapeError) -> String {
  let ShapeError(message) = self
  message
}

///|
pub(all) struct Shape {
  dimensions_ : Array[Int]
}

///|
pub fn Shape::new(dimensions : Array[Int]) -> Shape raise ShapeError {
  for index, dimension in dimensions {
    if dimension <= 0 {
      raise ShapeError(
        "dimension \{index} must be positive, received \{dimension}",
      )
    }
  }
  { dimensions_: dimensions.copy() }
}

///|
pub fn Shape::dimensions(self : Shape) -> Array[Int] {
  self.dimensions_.copy()
}

///|
pub fn Shape::rank(self : Shape) -> Int {
  self.dimensions_.length()
}

///|
pub fn Shape::dimension(self : Shape, index : Int) -> Int {
  self.dimensions_[index]
}

///|
pub fn Shape::to_string(self : Shape) -> String {
  let parts = self.dimensions_.map(fn(dimension) { dimension.to_string() })
  "[" + parts.join(", ") + "]"
}

///|
pub fn Shape::element_count(self : Shape) -> Int {
  self.dimensions_.fold(init=1, fn(count, dimension) { count * dimension })
}

///|
pub fn Shape::same_as(self : Shape, other : Shape) -> Bool {
  if self.rank() != other.rank() {
    return false
  }
  for index in 0..<self.rank() {
    if self.dimension(index) != other.dimension(index) {
      return false
    }
  }
  true
}

///|
pub fn Shape::broadcast(lhs : Shape, rhs : Shape) -> Shape raise ShapeError {
  let rank = if lhs.rank() > rhs.rank() { lhs.rank() } else { rhs.rank() }
  let dimensions = Array::make(rank, 1)
  for output_index in 0..<rank {
    let lhs_index = output_index - (rank - lhs.rank())
    let rhs_index = output_index - (rank - rhs.rank())
    let lhs_dimension = if lhs_index < 0 { 1 } else { lhs.dimension(lhs_index) }
    let rhs_dimension = if rhs_index < 0 { 1 } else { rhs.dimension(rhs_index) }
    if lhs_dimension != rhs_dimension &&
      lhs_dimension != 1 &&
      rhs_dimension != 1 {
      raise ShapeError(
        "cannot broadcast \{lhs.to_string()} and \{rhs.to_string()}",
      )
    }
    dimensions[output_index] = if lhs_dimension > rhs_dimension {
      lhs_dimension
    } else {
      rhs_dimension
    }
  }
  Shape::new(dimensions)
}

///|
pub fn Shape::matmul(lhs : Shape, rhs : Shape) -> Shape raise ShapeError {
  if lhs.rank() < 2 || rhs.rank() < 2 {
    raise ShapeError("matmul requires tensors with rank 2 or greater")
  }
  let lhs_rows = lhs.dimension(lhs.rank() - 2)
  let lhs_columns = lhs.dimension(lhs.rank() - 1)
  let rhs_rows = rhs.dimension(rhs.rank() - 2)
  let rhs_columns = rhs.dimension(rhs.rank() - 1)
  if lhs_columns != rhs_rows {
    raise ShapeError(
      "matmul inner dimensions differ: \{lhs_columns} and \{rhs_rows}",
    )
  }
  let lhs_batch_dimensions : Array[Int] = []
  for axis in 0..<(lhs.rank() - 2) {
    lhs_batch_dimensions.push(lhs.dimension(axis))
  }
  let rhs_batch_dimensions : Array[Int] = []
  for axis in 0..<(rhs.rank() - 2) {
    rhs_batch_dimensions.push(rhs.dimension(axis))
  }
  let batch_shape = Shape::broadcast(
    Shape::new(lhs_batch_dimensions),
    Shape::new(rhs_batch_dimensions),
  )
  let output_dimensions = batch_shape.dimensions()
  output_dimensions.push(lhs_rows)
  output_dimensions.push(rhs_columns)
  Shape::new(output_dimensions)
}

///|
/// Return the shape produced by binary concatenation along axis.
pub fn Shape::concat(
  lhs : Shape,
  rhs : Shape,
  axis : Int,
) -> Shape raise ShapeError {
  if lhs.rank() != rhs.rank() || axis < 0 || axis >= lhs.rank() {
    raise ShapeError(
      "cannot concatenate \{lhs.to_string()} and \{rhs.to_string()} along axis \{axis}",
    )
  }
  let dimensions = lhs.dimensions()
  for index in 0..<lhs.rank() {
    if index == axis {
      dimensions[index] = lhs.dimension(index) + rhs.dimension(index)
    } else if lhs.dimension(index) != rhs.dimension(index) {
      raise ShapeError(
        "cannot concatenate \{lhs.to_string()} and \{rhs.to_string()} along axis \{axis}",
      )
    }
  }
  Shape::new(dimensions)
}

///|
pub fn Shape::transpose(
  self : Shape,
  permutation : Array[Int],
) -> Shape raise ShapeError {
  if permutation.length() != self.rank() {
    raise ShapeError(
      "transpose permutation rank \{permutation.length()} does not match \{self.rank()}",
    )
  }
  let seen = Array::make(self.rank(), false)
  let dimensions = Array::make(self.rank(), 1)
  for output_index, input_index in permutation {
    if input_index < 0 || input_index >= self.rank() || seen[input_index] {
      let parts = permutation.map(fn(index) { index.to_string() })
      raise ShapeError(
        "invalid transpose permutation [" + parts.join(", ") + "]",
      )
    }
    seen[input_index] = true
    dimensions[output_index] = self.dimension(input_index)
  }
  Shape::new(dimensions)
}

///|
pub fn Shape::reduce(
  self : Shape,
  axes : Array[Int],
  keep_dimensions : Bool,
) -> Shape raise ShapeError {
  if axes.is_empty() {
    raise ShapeError("reduction axes must not be empty")
  }
  let reduced = Array::make(self.rank(), false)
  for axis in axes {
    if axis < 0 || axis >= self.rank() || reduced[axis] {
      raise ShapeError("invalid reduction axis \\{axis}")
    }
    reduced[axis] = true
  }
  let dimensions : Array[Int] = []
  for axis in 0..<self.rank() {
    if reduced[axis] {
      if keep_dimensions {
        dimensions.push(1)
      }
    } else {
      dimensions.push(self.dimension(axis))
    }
  }
  if dimensions.is_empty() {
    dimensions.push(1)
  }
  Shape::new(dimensions)
}

///|
pub fn Shape::gather(
  self : Shape,
  indices : Shape,
  axis : Int,
) -> Shape raise ShapeError {
  if axis < 0 || axis >= self.rank() {
    raise ShapeError("invalid gather axis \\{axis}")
  }
  let dimensions : Array[Int] = []
  for index in 0..<axis {
    dimensions.push(self.dimension(index))
  }
  for dimension in indices.dimensions() {
    dimensions.push(dimension)
  }
  for index in (axis + 1)..<self.rank() {
    dimensions.push(self.dimension(index))
  }
  Shape::new(dimensions)
}

///|
pub fn Shape::slice(
  self : Shape,
  starts : Array[Int],
  sizes : Array[Int],
) -> Shape raise ShapeError {
  if starts.length() != self.rank() || sizes.length() != self.rank() {
    raise ShapeError("slice starts and sizes must match input rank")
  }
  for axis in 0..<self.rank() {
    if starts[axis] < 0 ||
      sizes[axis] <= 0 ||
      starts[axis] + sizes[axis] > self.dimension(axis) {
      raise ShapeError("slice is outside input shape")
    }
  }
  Shape::new(sizes)
}

///|
pub fn Shape::layer_normalization(
  input : Shape,
  scale : Shape,
  bias : Shape,
  axes : Array[Int],
  epsilon : Float,
) -> Shape raise ShapeError {
  if axes.is_empty() {
    raise ShapeError("layer normalization axes must not be empty")
  }
  if !(epsilon > 0.0) {
    raise ShapeError("layer normalization epsilon must be positive")
  }
  let seen = Array::make(input.rank(), false)
  let parameter_dimensions : Array[Int] = []
  for axis in axes {
    if axis < 0 || axis >= input.rank() || seen[axis] {
      raise ShapeError("invalid layer normalization axis \{axis}")
    }
    seen[axis] = true
    parameter_dimensions.push(input.dimension(axis))
  }
  let parameter_shape = Shape::new(parameter_dimensions)
  if !scale.same_as(parameter_shape) {
    raise ShapeError(
      "layer normalization scale shape \{scale.to_string()} must match axes shape \{parameter_shape.to_string()}",
    )
  }
  if !bias.same_as(parameter_shape) {
    raise ShapeError(
      "layer normalization bias shape \{bias.to_string()} must match axes shape \{parameter_shape.to_string()}",
    )
  }
  input
}

///|
pub(all) enum InputLayout {
  Nchw
  Nhwc
} derive(Debug, Eq)

///|
pub fn InputLayout::to_webnn_string(self : InputLayout) -> String {
  match self {
    Nchw => "nchw"
    Nhwc => "nhwc"
  }
}

///|
pub(all) enum Conv2dFilterLayout {
  Oihw
  Hwio
} derive(Debug, Eq)

///|
pub fn Conv2dFilterLayout::to_webnn_string(self : Conv2dFilterLayout) -> String {
  match self {
    Oihw => "oihw"
    Hwio => "hwio"
  }
}

///|
/// Validated 2-D pooling options.
pub struct Pool2dOptions {
  window_dimensions_ : Array[Int]
  padding_ : Array[Int]
  strides_ : Array[Int]
  dilations_ : Array[Int]
  input_layout_ : InputLayout
}

///|
pub fn Pool2dOptions::new(
  window_dimensions : Array[Int],
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
) -> Pool2dOptions raise ShapeError {
  Pool2dOptions::new_with_layout(
    window_dimensions,
    padding,
    strides,
    dilations,
    InputLayout::Nchw,
  )
}

///|
pub fn Pool2dOptions::new_with_layout(
  window_dimensions : Array[Int],
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  input_layout : InputLayout,
) -> Pool2dOptions raise ShapeError {
  if window_dimensions.length() != 2 {
    raise ShapeError("pool2d window dimensions must contain two values")
  }
  if padding.length() != 4 {
    raise ShapeError("pool2d padding must contain four values")
  }
  if strides.length() != 2 {
    raise ShapeError("pool2d strides must contain two values")
  }
  if dilations.length() != 2 {
    raise ShapeError("pool2d dilations must contain two values")
  }
  for value in window_dimensions {
    if value <= 0 {
      raise ShapeError("pool2d window dimensions must be positive")
    }
  }
  for value in padding {
    if value < 0 {
      raise ShapeError("pool2d padding must be non-negative")
    }
  }
  for value in strides {
    if value <= 0 {
      raise ShapeError("pool2d strides must be positive")
    }
  }
  for value in dilations {
    if value <= 0 {
      raise ShapeError("pool2d dilations must be positive")
    }
  }
  {
    window_dimensions_: window_dimensions.copy(),
    padding_: padding.copy(),
    strides_: strides.copy(),
    dilations_: dilations.copy(),
    input_layout_: input_layout,
  }
}

///|
pub fn Pool2dOptions::window_dimensions(self : Pool2dOptions) -> Array[Int] {
  self.window_dimensions_.copy()
}

///|
pub fn Pool2dOptions::padding(self : Pool2dOptions) -> Array[Int] {
  self.padding_.copy()
}

///|
pub fn Pool2dOptions::strides(self : Pool2dOptions) -> Array[Int] {
  self.strides_.copy()
}

///|
pub fn Pool2dOptions::dilations(self : Pool2dOptions) -> Array[Int] {
  self.dilations_.copy()
}

///|
pub fn Pool2dOptions::input_layout(self : Pool2dOptions) -> InputLayout {
  self.input_layout_
}

///|
/// Validated 2-D convolution options.
pub struct Conv2dOptions {
  padding_ : Array[Int]
  strides_ : Array[Int]
  dilations_ : Array[Int]
  groups_ : Int
  input_layout_ : InputLayout
  filter_layout_ : Conv2dFilterLayout
}

///|
pub fn Conv2dOptions::new(
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  groups : Int,
) -> Conv2dOptions raise ShapeError {
  Conv2dOptions::new_with_layout(
    padding,
    strides,
    dilations,
    groups,
    InputLayout::Nchw,
    Conv2dFilterLayout::Oihw,
  )
}

///|
pub fn Conv2dOptions::new_with_layout(
  padding : Array[Int],
  strides : Array[Int],
  dilations : Array[Int],
  groups : Int,
  input_layout : InputLayout,
  filter_layout : Conv2dFilterLayout,
) -> Conv2dOptions raise ShapeError {
  if padding.length() != 4 {
    raise ShapeError("conv2d padding must contain four values")
  }
  if strides.length() != 2 {
    raise ShapeError("conv2d strides must contain two values")
  }
  if dilations.length() != 2 {
    raise ShapeError("conv2d dilations must contain two values")
  }
  for value in padding {
    if value < 0 {
      raise ShapeError("conv2d padding must be non-negative")
    }
  }
  for value in strides {
    if value <= 0 {
      raise ShapeError("conv2d strides must be positive")
    }
  }
  for value in dilations {
    if value <= 0 {
      raise ShapeError("conv2d dilations must be positive")
    }
  }
  if groups <= 0 {
    raise ShapeError("conv2d groups must be positive")
  }
  {
    padding_: padding.copy(),
    strides_: strides.copy(),
    dilations_: dilations.copy(),
    groups_: groups,
    input_layout_: input_layout,
    filter_layout_: filter_layout,
  }
}

///|
pub fn Conv2dOptions::default() -> Conv2dOptions {
  {
    padding_: [0, 0, 0, 0],
    strides_: [1, 1],
    dilations_: [1, 1],
    groups_: 1,
    input_layout_: InputLayout::Nchw,
    filter_layout_: Conv2dFilterLayout::Oihw,
  }
}

///|
pub fn Conv2dOptions::padding(self : Conv2dOptions) -> Array[Int] {
  self.padding_.copy()
}

///|
pub fn Conv2dOptions::strides(self : Conv2dOptions) -> Array[Int] {
  self.strides_.copy()
}

///|
pub fn Conv2dOptions::dilations(self : Conv2dOptions) -> Array[Int] {
  self.dilations_.copy()
}

///|
pub fn Conv2dOptions::groups(self : Conv2dOptions) -> Int {
  self.groups_
}

///|
pub fn Conv2dOptions::input_layout(self : Conv2dOptions) -> InputLayout {
  self.input_layout_
}

///|
pub fn Conv2dOptions::filter_layout(self : Conv2dOptions) -> Conv2dFilterLayout {
  self.filter_layout_
}

///|
pub fn Shape::conv2d_nchw_oihw(
  input : Shape,
  filter : Shape,
  options : Conv2dOptions,
) -> Shape raise ShapeError {
  if options.input_layout_ != InputLayout::Nchw ||
    options.filter_layout_ != Conv2dFilterLayout::Oihw {
    raise ShapeError("conv2d_nchw_oihw requires NCHW input and OIHW filter")
  }
  Shape::conv2d(input, filter, options)
}

///|
pub fn Shape::pool2d(
  input : Shape,
  options : Pool2dOptions,
) -> Shape raise ShapeError {
  if input.rank() != 4 {
    raise ShapeError("pool2d requires a rank-4 input")
  }
  let (batches, input_height, input_width, channels) = match
    options.input_layout_ {
    InputLayout::Nchw =>
      (
        input.dimension(0),
        input.dimension(2),
        input.dimension(3),
        input.dimension(1),
      )
    InputLayout::Nhwc =>
      (
        input.dimension(0),
        input.dimension(1),
        input.dimension(2),
        input.dimension(3),
      )
  }
  let effective_height = (options.window_dimensions_[0] - 1) *
    options.dilations_[0] +
    1
  let effective_width = (options.window_dimensions_[1] - 1) *
    options.dilations_[1] +
    1
  let height_numerator = input_height -
    effective_height +
    options.padding_[0] +
    options.padding_[1]
  let width_numerator = input_width -
    effective_width +
    options.padding_[2] +
    options.padding_[3]
  if height_numerator < 0 || width_numerator < 0 {
    raise ShapeError("pool2d effective window exceeds padded input")
  }
  let output_height = height_numerator / options.strides_[0] + 1
  let output_width = width_numerator / options.strides_[1] + 1
  match options.input_layout_ {
    InputLayout::Nchw =>
      Shape::new([batches, channels, output_height, output_width])
    InputLayout::Nhwc =>
      Shape::new([batches, output_height, output_width, channels])
  }
}

///|
pub fn Shape::conv2d(
  input : Shape,
  filter : Shape,
  options : Conv2dOptions,
) -> Shape raise ShapeError {
  if input.rank() != 4 || filter.rank() != 4 {
    raise ShapeError("conv2d requires rank-4 input and filter")
  }
  let (batches, input_height, input_width, input_channels) = match
    options.input_layout_ {
    InputLayout::Nchw =>
      (
        input.dimension(0),
        input.dimension(2),
        input.dimension(3),
        input.dimension(1),
      )
    InputLayout::Nhwc =>
      (
        input.dimension(0),
        input.dimension(1),
        input.dimension(2),
        input.dimension(3),
      )
  }
  let (filter_height, filter_width, filter_channels, output_channels) = match
    options.filter_layout_ {
    Conv2dFilterLayout::Oihw =>
      (
        filter.dimension(2),
        filter.dimension(3),
        filter.dimension(1),
        filter.dimension(0),
      )
    Conv2dFilterLayout::Hwio =>
      (
        filter.dimension(0),
        filter.dimension(1),
        filter.dimension(2),
        filter.dimension(3),
      )
  }
  if input_channels % options.groups_ != 0 {
    raise ShapeError("conv2d input channels must be divisible by groups")
  }
  if output_channels % options.groups_ != 0 {
    raise ShapeError("conv2d output channels must be divisible by groups")
  }
  if filter_channels != input_channels / options.groups_ {
    raise ShapeError(
      "conv2d filter input channels do not match input channels/groups",
    )
  }
  let effective_height = (filter_height - 1) * options.dilations_[0] + 1
  let effective_width = (filter_width - 1) * options.dilations_[1] + 1
  let height_numerator = input_height -
    effective_height +
    options.padding_[0] +
    options.padding_[1]
  let width_numerator = input_width -
    effective_width +
    options.padding_[2] +
    options.padding_[3]
  if height_numerator < 0 || width_numerator < 0 {
    raise ShapeError("conv2d effective filter exceeds padded input")
  }
  let output_height = height_numerator / options.strides_[0] + 1
  let output_width = width_numerator / options.strides_[1] + 1
  match options.input_layout_ {
    InputLayout::Nchw =>
      Shape::new([batches, output_channels, output_height, output_width])
    InputLayout::Nhwc =>
      Shape::new([batches, output_height, output_width, output_channels])
  }
}

///|
pub fn Shape::validate_axis(self : Shape, axis : Int) -> Unit raise ShapeError {
  if axis < 0 || axis >= self.rank() {
    raise ShapeError("axis \{axis} is outside rank \{self.rank()}")
  }
}
