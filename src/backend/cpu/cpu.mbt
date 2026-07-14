///|
pub struct CpuTensor {
  shape_ : @shape.Shape
  values_ : Array[Float]
}

///|
pub fn CpuTensor::new(
  shape : @shape.Shape,
  values : Array[Float],
) -> CpuTensor raise @tensor.TensorError {
  if shape.element_count() != values.length() {
    raise @tensor.TensorError::new(
      "data length \{values.length()} does not match shape \{shape.to_string()}",
    )
  }
  { shape_: shape, values_: values.copy() }
}

///|
pub fn CpuTensor::values(self : CpuTensor) -> Array[Float] {
  self.values_.copy()
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
fn strides(shape : @shape.Shape) -> Array[Int] {
  let result = Array::make(shape.rank(), 1)
  let mut stride = 1
  for reverse_index in 0..<shape.rank() {
    let index = shape.rank() - reverse_index - 1
    result[index] = stride
    stride = stride * shape.dimension(index)
  }
  result
}

///|
fn broadcast_flat_index(
  output_index : Int,
  output_shape : @shape.Shape,
  input_shape : @shape.Shape,
) -> Int {
  let output_strides = strides(output_shape)
  let input_strides = strides(input_shape)
  let rank_offset = output_shape.rank() - input_shape.rank()
  let mut input_index = 0
  for output_axis in 0..<output_shape.rank() {
    let coordinate = output_index /
      output_strides[output_axis] %
      output_shape.dimension(output_axis)
    let input_axis = output_axis - rank_offset
    if input_axis >= 0 && input_shape.dimension(input_axis) != 1 {
      input_index = input_index + coordinate * input_strides[input_axis]
    }
  }
  input_index
}

///|
fn normalization_group_index(
  input_index : Int,
  input_shape : @shape.Shape,
  input_strides : Array[Int],
  normalized_axes : Array[Bool],
) -> Int {
  let mut group_index = 0
  for axis in 0..<input_shape.rank() {
    if !normalized_axes[axis] {
      let coordinate = input_index /
        input_strides[axis] %
        input_shape.dimension(axis)
      group_index = group_index * input_shape.dimension(axis) + coordinate
    }
  }
  group_index
}

///|
fn normalization_parameter_index(
  input_index : Int,
  input_shape : @shape.Shape,
  input_strides : Array[Int],
  parameter_strides : Array[Int],
  axes : Array[Int],
) -> Int {
  let mut parameter_index = 0
  for parameter_axis, input_axis in axes {
    let coordinate = input_index /
      input_strides[input_axis] %
      input_shape.dimension(input_axis)
    parameter_index = parameter_index +
      coordinate * parameter_strides[parameter_axis]
  }
  parameter_index
}

///|
fn broadcast_matrix_offset(
  output_batch : Int,
  output_shape : @shape.Shape,
  input_shape : @shape.Shape,
) -> Int {
  let output_batch_rank = output_shape.rank() - 2
  let input_batch_rank = input_shape.rank() - 2
  let output_matrix_size = output_shape.dimension(output_shape.rank() - 2) *
    output_shape.dimension(output_shape.rank() - 1)
  let output_flat_index = output_batch * output_matrix_size
  let output_strides = strides(output_shape)
  let input_strides = strides(input_shape)
  let rank_offset = output_batch_rank - input_batch_rank
  let mut input_offset = 0
  for output_axis in 0..<output_batch_rank {
    let input_axis = output_axis - rank_offset
    if input_axis >= 0 && input_shape.dimension(input_axis) != 1 {
      let coordinate = output_flat_index /
        output_strides[output_axis] %
        output_shape.dimension(output_axis)
      input_offset = input_offset + coordinate * input_strides[input_axis]
    }
  }
  input_offset
}

///|
fn elementwise(
  lhs : CpuTensor,
  rhs : CpuTensor,
  operation : (Float, Float) -> Float,
) -> CpuTensor raise @tensor.TensorError {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::broadcast(lhs.shape_, rhs.shape_)
  })
  let values : Array[Float] = Array::make(
    output_shape.element_count(),
    Float::from_int(0),
  )
  for output_index in 0..<values.length() {
    let lhs_index = broadcast_flat_index(output_index, output_shape, lhs.shape_)
    let rhs_index = broadcast_flat_index(output_index, output_shape, rhs.shape_)
    values[output_index] = operation(
      lhs.values_[lhs_index],
      rhs.values_[rhs_index],
    )
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn shape(self : CpuTensor) -> @shape.Shape {
  self.shape_
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn add(
  lhs : CpuTensor,
  rhs : CpuTensor,
) -> CpuTensor {
  elementwise(lhs, rhs, fn(a, b) { a + b })
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn sub(
  lhs : CpuTensor,
  rhs : CpuTensor,
) -> CpuTensor {
  elementwise(lhs, rhs, fn(a, b) { a - b })
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn mul(
  lhs : CpuTensor,
  rhs : CpuTensor,
) -> CpuTensor {
  elementwise(lhs, rhs, fn(a, b) { a * b })
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn div(
  lhs : CpuTensor,
  rhs : CpuTensor,
) -> CpuTensor {
  elementwise(lhs, rhs, fn(a, b) { a / b })
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn reduce_mean(
  self : CpuTensor,
  axes : Array[Int],
  keep_dimensions : Bool,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    self.shape_.reduce(axes, keep_dimensions)
  })
  let reduced = Array::make(self.shape_.rank(), false)
  let mut count = 1
  for axis in axes {
    reduced[axis] = true
    count = count * self.shape_.dimension(axis)
  }
  let input_strides = strides(self.shape_)
  let output_strides = strides(output_shape)
  let values = Array::make(output_shape.element_count(), Float::from_int(0))
  for input_index in 0..<self.values_.length() {
    let mut output_index = 0
    let mut output_axis = 0
    for axis in 0..<self.shape_.rank() {
      if !reduced[axis] {
        let coordinate = input_index /
          input_strides[axis] %
          self.shape_.dimension(axis)
        if keep_dimensions {
          output_index = output_index + coordinate * output_strides[axis]
        } else {
          output_index = output_index + coordinate * output_strides[output_axis]
        }
      }
      if !keep_dimensions && !reduced[axis] {
        output_axis = output_axis + 1
      }
    }
    values[output_index] = values[output_index] + self.values_[input_index]
  }
  let divisor = Float::from_int(count)
  for index in 0..<values.length() {
    values[index] = values[index] / divisor
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn gather(
  self : CpuTensor,
  indices : Array[Int],
  indices_shape : @shape.Shape,
  axis : Int,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    self.shape_.gather(indices_shape, axis)
  })
  if indices.length() != indices_shape.element_count() {
    raise @tensor.TensorError::new("gather indices length does not match shape")
  }
  let mut outer = 1
  for dimension in 0..<axis {
    outer = outer * self.shape_.dimension(dimension)
  }
  let mut inner = 1
  for dimension in (axis + 1)..<self.shape_.rank() {
    inner = inner * self.shape_.dimension(dimension)
  }
  let axis_size = self.shape_.dimension(axis)
  let values : Array[Float] = []
  for outer_index in 0..<outer {
    for index in indices {
      if index < 0 || index >= axis_size {
        raise @tensor.TensorError::new("gather index is outside axis")
      }
      let offset = (outer_index * axis_size + index) * inner
      for inner_index in 0..<inner {
        values.push(self.values_[offset + inner_index])
      }
    }
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn slice(
  self : CpuTensor,
  starts : Array[Int],
  sizes : Array[Int],
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    self.shape_.slice(starts, sizes)
  })
  let input_strides = strides(self.shape_)
  let output_strides = strides(output_shape)
  let values : Array[Float] = []
  for output_index in 0..<output_shape.element_count() {
    let mut input_index = 0
    for axis in 0..<self.shape_.rank() {
      let coordinate = output_index /
        output_strides[axis] %
        output_shape.dimension(axis)
      input_index = input_index +
        (starts[axis] + coordinate) * input_strides[axis]
    }
    values.push(self.values_[input_index])
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn concat(
  lhs : CpuTensor,
  rhs : CpuTensor,
  axis : Int,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::concat(lhs.shape_, rhs.shape_, axis)
  })
  let mut outer = 1
  for index in 0..<axis {
    outer = outer * lhs.shape_.dimension(index)
  }
  let mut inner = 1
  for index in (axis + 1)..<lhs.shape_.rank() {
    inner = inner * lhs.shape_.dimension(index)
  }
  let lhs_chunk = lhs.shape_.dimension(axis) * inner
  let rhs_chunk = rhs.shape_.dimension(axis) * inner
  let values : Array[Float] = []
  for group in 0..<outer {
    let lhs_offset = group * lhs_chunk
    let rhs_offset = group * rhs_chunk
    for index in 0..<lhs_chunk {
      values.push(lhs.values_[lhs_offset + index])
    }
    for index in 0..<rhs_chunk {
      values.push(rhs.values_[rhs_offset + index])
    }
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn matmul(
  lhs : CpuTensor,
  rhs : CpuTensor,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::matmul(lhs.shape_, rhs.shape_)
  })
  let rows = lhs.shape_.dimension(lhs.shape_.rank() - 2)
  let inner = lhs.shape_.dimension(lhs.shape_.rank() - 1)
  let columns = rhs.shape_.dimension(rhs.shape_.rank() - 1)
  let matrix_size = rows * columns
  let batch_count = output_shape.element_count() / matrix_size
  let values : Array[Float] = Array::make(
    output_shape.element_count(),
    Float::from_int(0),
  )
  for batch in 0..<batch_count {
    let lhs_offset = broadcast_matrix_offset(batch, output_shape, lhs.shape_)
    let rhs_offset = broadcast_matrix_offset(batch, output_shape, rhs.shape_)
    let output_offset = batch * matrix_size
    for row in 0..<rows {
      for column in 0..<columns {
        let mut sum : Float = Float::from_int(0)
        for index in 0..<inner {
          sum = sum +
            lhs.values_[lhs_offset + row * inner + index] *
            rhs.values_[rhs_offset + index * columns + column]
        }
        values[output_offset + row * columns + column] = sum
      }
    }
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn conv2d(
  input : CpuTensor,
  filter : CpuTensor,
  options : @shape.Conv2dOptions,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::conv2d(input.shape_, filter.shape_, options)
  })
  let batches = input.shape_.dimension(0)
  let input_layout = options.input_layout()
  let filter_layout = options.filter_layout()
  let (input_height, input_width, input_channels) = match input_layout {
    @shape.InputLayout::Nchw =>
      (
        input.shape_.dimension(2),
        input.shape_.dimension(3),
        input.shape_.dimension(1),
      )
    @shape.InputLayout::Nhwc =>
      (
        input.shape_.dimension(1),
        input.shape_.dimension(2),
        input.shape_.dimension(3),
      )
  }
  let (output_height, output_width, output_channels) = match input_layout {
    @shape.InputLayout::Nchw =>
      (
        output_shape.dimension(2),
        output_shape.dimension(3),
        output_shape.dimension(1),
      )
    @shape.InputLayout::Nhwc =>
      (
        output_shape.dimension(1),
        output_shape.dimension(2),
        output_shape.dimension(3),
      )
  }
  let (filter_height, filter_width, filter_channels) = match filter_layout {
    @shape.Conv2dFilterLayout::Oihw =>
      (
        filter.shape_.dimension(2),
        filter.shape_.dimension(3),
        filter.shape_.dimension(1),
      )
    @shape.Conv2dFilterLayout::Hwio =>
      (
        filter.shape_.dimension(0),
        filter.shape_.dimension(1),
        filter.shape_.dimension(2),
      )
  }
  let padding = options.padding()
  let conv_strides = options.strides()
  let dilations = options.dilations()
  let output_channels_per_group = output_channels / options.groups()
  let values = Array::make(output_shape.element_count(), Float::from_int(0))
  for batch in 0..<batches {
    for output_channel in 0..<output_channels {
      let group = output_channel / output_channels_per_group
      let input_channel_start = group * filter_channels
      for output_y in 0..<output_height {
        for output_x in 0..<output_width {
          let mut sum = Float::from_int(0)
          for filter_channel in 0..<filter_channels {
            let input_channel = input_channel_start + filter_channel
            for filter_y in 0..<filter_height {
              let input_y = output_y * conv_strides[0] -
                padding[0] +
                filter_y * dilations[0]
              if input_y < 0 || input_y >= input_height {
                continue
              }
              for filter_x in 0..<filter_width {
                let input_x = output_x * conv_strides[1] -
                  padding[2] +
                  filter_x * dilations[1]
                if input_x < 0 || input_x >= input_width {
                  continue
                }
                let input_index = match input_layout {
                  @shape.InputLayout::Nchw =>
                    (
                      (batch * input_channels + input_channel) * input_height +
                      input_y
                    ) *
                    input_width +
                    input_x
                  @shape.InputLayout::Nhwc =>
                    ((batch * input_height + input_y) * input_width + input_x) *
                    input_channels +
                    input_channel
                }
                let filter_index = match filter_layout {
                  @shape.Conv2dFilterLayout::Oihw =>
                    (
                      (output_channel * filter_channels + filter_channel) *
                      filter_height +
                      filter_y
                    ) *
                    filter_width +
                    filter_x
                  @shape.Conv2dFilterLayout::Hwio =>
                    (
                      (filter_y * filter_width + filter_x) * filter_channels +
                      filter_channel
                    ) *
                    output_channels +
                    output_channel
                }
                sum = sum +
                  input.values_[input_index] * filter.values_[filter_index]
              }
            }
          }
          let output_index = match input_layout {
            @shape.InputLayout::Nchw =>
              (
                (batch * output_channels + output_channel) * output_height +
                output_y
              ) *
              output_width +
              output_x
            @shape.InputLayout::Nhwc =>
              ((batch * output_height + output_y) * output_width + output_x) *
              output_channels +
              output_channel
          }
          values[output_index] = sum
        }
      }
    }
  }
  CpuTensor::new(output_shape, values)
}

///|
fn pool2d(
  input : CpuTensor,
  options : @shape.Pool2dOptions,
  maximum : Bool,
) -> CpuTensor raise @tensor.TensorError {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::pool2d(input.shape_, options)
  })
  let input_layout = options.input_layout()
  let (batches, input_height, input_width, channels) = match input_layout {
    @shape.InputLayout::Nchw =>
      (
        input.shape_.dimension(0),
        input.shape_.dimension(2),
        input.shape_.dimension(3),
        input.shape_.dimension(1),
      )
    @shape.InputLayout::Nhwc =>
      (
        input.shape_.dimension(0),
        input.shape_.dimension(1),
        input.shape_.dimension(2),
        input.shape_.dimension(3),
      )
  }
  let (output_height, output_width) = match input_layout {
    @shape.InputLayout::Nchw =>
      (output_shape.dimension(2), output_shape.dimension(3))
    @shape.InputLayout::Nhwc =>
      (output_shape.dimension(1), output_shape.dimension(2))
  }
  let window_dimensions = options.window_dimensions()
  let padding = options.padding()
  let pool_strides = options.strides()
  let dilations = options.dilations()
  let values = Array::make(output_shape.element_count(), Float::from_int(0))
  for batch in 0..<batches {
    for channel in 0..<channels {
      for output_y in 0..<output_height {
        for output_x in 0..<output_width {
          let mut sum = Float::from_int(0)
          let mut largest = Float::from_int(0)
          let mut has_value = false
          let mut count = 0
          for window_y in 0..<window_dimensions[0] {
            let input_y = output_y * pool_strides[0] -
              padding[0] +
              window_y * dilations[0]
            if input_y < 0 || input_y >= input_height {
              continue
            }
            for window_x in 0..<window_dimensions[1] {
              let input_x = output_x * pool_strides[1] -
                padding[2] +
                window_x * dilations[1]
              if input_x < 0 || input_x >= input_width {
                continue
              }
              let input_index = match input_layout {
                @shape.InputLayout::Nchw =>
                  ((batch * channels + channel) * input_height + input_y) *
                  input_width +
                  input_x
                @shape.InputLayout::Nhwc =>
                  ((batch * input_height + input_y) * input_width + input_x) *
                  channels +
                  channel
              }
              let value = input.values_[input_index]
              sum = sum + value
              if !has_value || value > largest {
                largest = value
              }
              has_value = true
              count = count + 1
            }
          }
          let output_index = match input_layout {
            @shape.InputLayout::Nchw =>
              ((batch * channels + channel) * output_height + output_y) *
              output_width +
              output_x
            @shape.InputLayout::Nhwc =>
              ((batch * output_height + output_y) * output_width + output_x) *
              channels +
              channel
          }
          values[output_index] = if maximum {
            largest
          } else {
            sum / Float::from_int(count)
          }
        }
      }
    }
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn max_pool2d(
  self : CpuTensor,
  options : @shape.Pool2dOptions,
) -> CpuTensor {
  pool2d(self, options, true)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn average_pool2d(
  self : CpuTensor,
  options : @shape.Pool2dOptions,
) -> CpuTensor {
  pool2d(self, options, false)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn sigmoid(self : CpuTensor) -> CpuTensor {
  let values = self.values_.map(fn(value) {
    if value >= 0.0 {
      Float::from_double(1.0 / (1.0 + @math.exp((-value).to_double())))
    } else {
      let exponential = @math.exp(value.to_double())
      Float::from_double(exponential / (1.0 + exponential))
    }
  })
  { shape_: self.shape_, values_: values }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn tanh(self : CpuTensor) -> CpuTensor {
  let values = self.values_.map(fn(value) {
    Float::from_double(@math.tanh(value.to_double()))
  })
  { shape_: self.shape_, values_: values }
}

///|
fn error_function(value : Double) -> Double {
  let magnitude = value.abs()
  let t = 1.0 / (1.0 + 0.5 * magnitude)
  let exponent = -magnitude * magnitude -
    1.26551223 +
    t *
    (
      1.00002368 +
      t *
      (
        0.37409196 +
        t *
        (
          0.09678418 +
          t *
          (
            -0.18628806 +
            t *
            (
              0.27886807 +
              t *
              (
                -1.13520398 +
                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
              )
            )
          )
        )
      )
    )
  let complement = t * @math.exp(exponent)
  if value >= 0.0 {
    1.0 - complement
  } else {
    complement - 1.0
  }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn gelu(self : CpuTensor) -> CpuTensor {
  let values = self.values_.map(fn(value) {
    let x = value.to_double()
    Float::from_double(0.5 * x * (1.0 + error_function(x / 1.4142135623730951)))
  })
  { shape_: self.shape_, values_: values }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn layer_normalization(
  input : CpuTensor,
  scale : CpuTensor,
  bias : CpuTensor,
  axes : Array[Int],
  epsilon : Float,
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    @shape.Shape::layer_normalization(
      input.shape_,
      scale.shape_,
      bias.shape_,
      axes,
      epsilon,
    )
  })
  let input_strides = strides(input.shape_)
  let parameter_strides = strides(scale.shape_)
  let normalized_axes = Array::make(input.shape_.rank(), false)
  for axis in axes {
    normalized_axes[axis] = true
  }
  let group_count = input.shape_.element_count() / scale.shape_.element_count()
  let means = Array::make(group_count, Float::from_int(0))
  for input_index in 0..<input.values_.length() {
    let group_index = normalization_group_index(
      input_index,
      input.shape_,
      input_strides,
      normalized_axes,
    )
    means[group_index] = means[group_index] + input.values_[input_index]
  }
  let normalized_count = Float::from_int(scale.shape_.element_count())
  for group_index in 0..<group_count {
    means[group_index] = means[group_index] / normalized_count
  }
  let variances = Array::make(group_count, Float::from_int(0))
  for input_index in 0..<input.values_.length() {
    let group_index = normalization_group_index(
      input_index,
      input.shape_,
      input_strides,
      normalized_axes,
    )
    let centered = input.values_[input_index] - means[group_index]
    variances[group_index] = variances[group_index] + centered * centered
  }
  for group_index in 0..<group_count {
    variances[group_index] = variances[group_index] / normalized_count
  }
  let values = Array::make(input.values_.length(), Float::from_int(0))
  for input_index in 0..<input.values_.length() {
    let group_index = normalization_group_index(
      input_index,
      input.shape_,
      input_strides,
      normalized_axes,
    )
    let parameter_index = normalization_parameter_index(
      input_index,
      input.shape_,
      input_strides,
      parameter_strides,
      axes,
    )
    let denominator = (variances[group_index] + epsilon).sqrt()
    let normalized = (input.values_[input_index] - means[group_index]) /
      denominator
    values[input_index] = normalized * scale.values_[parameter_index] +
      bias.values_[parameter_index]
  }
  CpuTensor::new(output_shape, values)
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn clamp(
  self : CpuTensor,
  minimum : Float,
  maximum : Float,
) -> CpuTensor {
  if minimum > maximum {
    raise @tensor.TensorError::new("clamp minimum must not exceed maximum")
  }
  let values = self.values_.map(fn(value) {
    if value < minimum {
      minimum
    } else if value > maximum {
      maximum
    } else {
      value
    }
  })
  { shape_: self.shape_, values_: values }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn relu(self : CpuTensor) -> CpuTensor {
  let values = self.values_.map(fn(value) {
    if value < Float::from_int(0) {
      Float::from_int(0)
    } else {
      value
    }
  })
  { shape_: self.shape_, values_: values }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn softmax(
  self : CpuTensor,
  axis : Int,
) -> CpuTensor {
  self.shape_.validate_axis(axis) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  let axis_size = self.shape_.dimension(axis)
  let mut inner = 1
  for index in (axis + 1)..<self.shape_.rank() {
    inner = inner * self.shape_.dimension(index)
  }
  let outer = self.shape_.element_count() / (axis_size * inner)
  let values = Array::make(self.values_.length(), Float::from_int(0))
  for outer_index in 0..<outer {
    for inner_index in 0..<inner {
      let first_index = outer_index * axis_size * inner + inner_index
      let mut maximum = self.values_[first_index]
      for axis_index in 1..<axis_size {
        let index = first_index + axis_index * inner
        if self.values_[index] > maximum {
          maximum = self.values_[index]
        }
      }
      let mut sum = 0.0
      for axis_index in 0..<axis_size {
        let index = first_index + axis_index * inner
        sum = sum + @math.exp((self.values_[index] - maximum).to_double())
      }
      for axis_index in 0..<axis_size {
        let index = first_index + axis_index * inner
        values[index] = Float::from_double(
          @math.exp((self.values_[index] - maximum).to_double()) / sum,
        )
      }
    }
  }
  { shape_: self.shape_, values_: values }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn reshape(
  self : CpuTensor,
  shape : @shape.Shape,
) -> CpuTensor {
  if self.shape_.element_count() != shape.element_count() {
    raise @tensor.TensorError::new(
      "reshape changes element count from \{self.shape_.element_count()} to \{shape.element_count()}",
    )
  }
  { shape_: shape, values_: self.values_.copy() }
}

///|
pub impl @tensor.TensorOps for CpuTensor with fn transpose(
  self : CpuTensor,
  permutation : Array[Int],
) -> CpuTensor {
  let output_shape = shape_or_tensor_error(() => {
    self.shape_.transpose(permutation)
  })
  let input_strides = strides(self.shape_)
  let output_strides = strides(output_shape)
  let values : Array[Float] = Array::make(
    self.values_.length(),
    Float::from_int(0),
  )
  for output_index in 0..<values.length() {
    let mut input_index = 0
    for output_axis in 0..<output_shape.rank() {
      let coordinate = output_index /
        output_strides[output_axis] %
        output_shape.dimension(output_axis)
      input_index = input_index +
        coordinate * input_strides[permutation[output_axis]]
    }
    values[output_index] = self.values_[input_index]
  }
  CpuTensor::new(output_shape, values)
}
