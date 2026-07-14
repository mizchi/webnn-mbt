///|
pub(all) struct TfliteModel {}

///|
pub enum TfliteRuntimeDataType {
  Float32
  UInt8(Array[Float], Array[Int], Int)
  Int8(Array[Float], Array[Int], Int)
}

///|
pub struct TfliteRuntimeTensor {
  name_ : String
  shape_ : @shape.Shape
  data_type_ : TfliteRuntimeDataType
}

///|
pub struct TfliteRuntimeModel {
  model_ : LiteRtModel
  inputs_ : Array[TfliteRuntimeTensor]
  outputs_ : Array[TfliteRuntimeTensor]
}

///|
priv struct TfliteTensor {
  name : String
  shape : @shape.Shape
  tensor_type : Int
  buffer_index : Int
  quantization : TfliteQuantization?
}

///|
priv struct TfliteQuantization {
  scales : Array[Float]
  zero_points : Array[Int]
  quantized_dimension : Int
}

///|
priv struct ParsedTfliteModel {
  model : LiteRtModel
  inputs : Array[TfliteTensor]
  outputs : Array[TfliteTensor]
}

///|
priv struct TfliteLoweredOperator {
  nodes : Array[LiteRtNode]
  values : Array[LiteRtValue]
}

///|
priv struct TfliteOperatorCode {
  builtin : Int
  custom : String?
}

///|
fn required_vector(
  table : FlatBufferTable,
  field_index : Int,
  element_size : Int,
  label : String,
) -> FlatBufferVector raise LiteRtError {
  match table.vector(field_index, element_size) {
    Some(vector) => vector
    None => raise LiteRtError("TFLite \{label} is missing")
  }
}

///|
fn vector_i32(vector : FlatBufferVector) -> Array[Int] raise LiteRtError {
  Array::makei(vector.length, fn(index) raise LiteRtError { vector.i32(index) })
}

///|
fn vector_f32(vector : FlatBufferVector) -> Array[Float] raise LiteRtError {
  if vector.element_size != 4 {
    raise LiteRtError("TFLite FlatBuffer float vector element size is invalid")
  }
  Array::makei(vector.length, fn(index) raise LiteRtError {
    Float::reinterpret_from_uint(vector.reader.u32(vector.position + index * 4))
  })
}

///|
fn vector_i64_as_int(vector : FlatBufferVector) -> Array[Int] raise LiteRtError {
  if vector.element_size != 8 {
    raise LiteRtError("TFLite FlatBuffer int64 vector element size is invalid")
  }
  Array::makei(vector.length, fn(index) raise LiteRtError {
    vector.reader.i64_as_int(vector.position + index * 8)
  })
}

///|
fn table_at(
  vector : FlatBufferVector,
  index : Int,
  label : String,
) -> FlatBufferTable raise LiteRtError {
  if index < 0 || index >= vector.length {
    raise LiteRtError("TFLite \{label} index is outside its vector")
  }
  vector.table(index)
}

///|
fn tensor_name(index : Int) -> String {
  "tensor_\{index}"
}

///|
fn unique_tflite_tensor_names(
  tensors : Array[TfliteTensor],
) -> Array[TfliteTensor] {
  let used : Map[String, Bool] = Map([])
  let unique : Array[TfliteTensor] = []
  for index, tensor in tensors {
    let mut name = tensor.name
    if used.contains(name) {
      name = tensor_name(index)
      let mut suffix = 1
      while used.contains(name) {
        name = tensor_name(index) + "_" + suffix.to_string()
        suffix = suffix + 1
      }
    }
    used[name] = true
    unique.push({
      name,
      shape: tensor.shape,
      tensor_type: tensor.tensor_type,
      buffer_index: tensor.buffer_index,
      quantization: tensor.quantization,
    })
  }
  unique
}

///|
fn validate_quantization(
  scale : Float,
  zero_point : Int,
  minimum : Int,
  maximum : Int,
) -> Unit raise LiteRtError {
  if scale <= 0.0 || zero_point < minimum || zero_point > maximum {
    raise LiteRtError("TFLite quantization parameters are invalid")
  }
}

///|
fn round_to_int(value : Float) -> Int {
  if value >= 0.0 {
    (value + 0.5).to_int()
  } else {
    (value - 0.5).to_int()
  }
}

///|
fn requantize(
  values : Array[Float],
  scale : Float,
  zero_point : Int,
  minimum : Int,
  maximum : Int,
) -> Array[Int] raise LiteRtError {
  validate_quantization(scale, zero_point, minimum, maximum)
  values.map(fn(value) {
    let quantized = round_to_int(value / scale) + zero_point
    if quantized < minimum {
      minimum
    } else if quantized > maximum {
      maximum
    } else {
      quantized
    }
  })
}

///|
/// Convert unsigned 8-bit affine values to their float32 values.
pub fn dequantize_uint8(
  values : Array[Int],
  scale : Float,
  zero_point : Int,
) -> Array[Float] raise LiteRtError {
  validate_quantization(scale, zero_point, 0, 255)
  values.map(fn(value) raise LiteRtError {
    if value < 0 || value > 255 {
      raise LiteRtError("UINT8 value is outside [0, 255]")
    }
    scale * Float::from_int(value - zero_point)
  })
}

///|
/// Convert float32 values to unsigned 8-bit affine values with saturation.
pub fn requantize_uint8(
  values : Array[Float],
  scale : Float,
  zero_point : Int,
) -> Array[Int] raise LiteRtError {
  requantize(values, scale, zero_point, 0, 255)
}

///|
/// Convert signed 8-bit affine values to their float32 values.
pub fn dequantize_int8(
  values : Array[Int],
  scale : Float,
  zero_point : Int,
) -> Array[Float] raise LiteRtError {
  validate_quantization(scale, zero_point, -128, 127)
  values.map(fn(value) raise LiteRtError {
    if value < -128 || value > 127 {
      raise LiteRtError("INT8 value is outside [-128, 127]")
    }
    scale * Float::from_int(value - zero_point)
  })
}

///|
/// Convert float32 values to signed 8-bit affine values with saturation.
pub fn requantize_int8(
  values : Array[Float],
  scale : Float,
  zero_point : Int,
) -> Array[Int] raise LiteRtError {
  requantize(values, scale, zero_point, -128, 127)
}

///|
fn validate_runtime_quantization(
  shape : @shape.Shape,
  scales : Array[Float],
  zero_points : Array[Int],
  quantized_dimension : Int,
  minimum : Int,
  maximum : Int,
) -> Unit raise LiteRtError {
  if scales.is_empty() ||
    (zero_points.length() != 1 && zero_points.length() != scales.length()) {
    raise LiteRtError("TFLite runtime quantization parameters are incompatible")
  }
  if scales.length() > 1 &&
    (
      quantized_dimension < 0 ||
      quantized_dimension >= shape.rank() ||
      scales.length() != shape.dimension(quantized_dimension)
    ) {
    raise LiteRtError(
      "TFLite runtime per-axis quantization is incompatible with tensor shape",
    )
  }
  for index in 0..<scales.length() {
    let zero_point_index = if zero_points.length() == 1 { 0 } else { index }
    validate_quantization(
      scales[index],
      zero_points[zero_point_index],
      minimum,
      maximum,
    )
  }
}

///|
fn quantized_axis_index(
  shape : @shape.Shape,
  scale_count : Int,
  quantized_dimension : Int,
  value_index : Int,
) -> Int {
  if scale_count == 1 {
    0
  } else {
    let mut stride = 1
    for dimension in (quantized_dimension + 1)..<shape.rank() {
      stride = stride * shape.dimension(dimension)
    }
    value_index / stride % shape.dimension(quantized_dimension)
  }
}

///|
fn dequantize_runtime_values(
  values : Array[Int],
  shape : @shape.Shape,
  scales : Array[Float],
  zero_points : Array[Int],
  quantized_dimension : Int,
  minimum : Int,
  maximum : Int,
  value_type : String,
) -> Array[Float] raise LiteRtError {
  validate_runtime_quantization(
    shape, scales, zero_points, quantized_dimension, minimum, maximum,
  )
  Array::makei(values.length(), fn(index) raise LiteRtError {
    let value = values[index]
    if value < minimum || value > maximum {
      raise LiteRtError(
        "TFLite runtime \{value_type} value is outside its range",
      )
    }
    let quantized_index = quantized_axis_index(
      shape,
      scales.length(),
      quantized_dimension,
      index,
    )
    let zero_point_index = if zero_points.length() == 1 {
      0
    } else {
      quantized_index
    }
    scales[quantized_index] *
    Float::from_int(value - zero_points[zero_point_index])
  })
}

///|
fn requantize_runtime_values(
  values : Array[Float],
  shape : @shape.Shape,
  scales : Array[Float],
  zero_points : Array[Int],
  quantized_dimension : Int,
  minimum : Int,
  maximum : Int,
) -> Array[Int] raise LiteRtError {
  validate_runtime_quantization(
    shape, scales, zero_points, quantized_dimension, minimum, maximum,
  )
  Array::makei(values.length(), fn(index) {
    let quantized_index = quantized_axis_index(
      shape,
      scales.length(),
      quantized_dimension,
      index,
    )
    let zero_point_index = if zero_points.length() == 1 {
      0
    } else {
      quantized_index
    }
    let quantized = round_to_int(values[index] / scales[quantized_index]) +
      zero_points[zero_point_index]
    if quantized < minimum {
      minimum
    } else if quantized > maximum {
      maximum
    } else {
      quantized
    }
  })
}

///|
pub fn TfliteRuntimeTensor::name(self : TfliteRuntimeTensor) -> String {
  self.name_
}

///|
pub fn TfliteRuntimeTensor::shape(self : TfliteRuntimeTensor) -> @shape.Shape {
  self.shape_
}

///|
pub fn TfliteRuntimeTensor::data_type(
  self : TfliteRuntimeTensor,
) -> TfliteRuntimeDataType {
  self.data_type_
}

///|
/// Dequantize raw runtime values according to this tensor's TFLite metadata.
pub fn TfliteRuntimeTensor::dequantize(
  self : TfliteRuntimeTensor,
  values : Array[Int],
) -> Array[Float] raise LiteRtError {
  if values.length() != self.shape_.element_count() {
    raise LiteRtError(
      "TFLite runtime input value count does not match its shape",
    )
  }
  match self.data_type_ {
    Float32 =>
      raise LiteRtError("TFLite runtime tensor is FLOAT32, not quantized")
    UInt8(scales, zero_points, quantized_dimension) =>
      dequantize_runtime_values(
        values,
        self.shape_,
        scales,
        zero_points,
        quantized_dimension,
        0,
        255,
        "UINT8",
      )
    Int8(scales, zero_points, quantized_dimension) =>
      dequantize_runtime_values(
        values,
        self.shape_,
        scales,
        zero_points,
        quantized_dimension,
        -128,
        127,
        "INT8",
      )
  }
}

///|
/// Requantize float32 backend results according to this tensor's TFLite metadata.
pub fn TfliteRuntimeTensor::requantize(
  self : TfliteRuntimeTensor,
  values : Array[Float],
) -> Array[Int] raise LiteRtError {
  if values.length() != self.shape_.element_count() {
    raise LiteRtError(
      "TFLite runtime output value count does not match its shape",
    )
  }
  match self.data_type_ {
    Float32 =>
      raise LiteRtError("TFLite runtime tensor is FLOAT32, not quantized")
    UInt8(scales, zero_points, quantized_dimension) =>
      requantize_runtime_values(
        values,
        self.shape_,
        scales,
        zero_points,
        quantized_dimension,
        0,
        255,
      )
    Int8(scales, zero_points, quantized_dimension) =>
      requantize_runtime_values(
        values,
        self.shape_,
        scales,
        zero_points,
        quantized_dimension,
        -128,
        127,
      )
  }
}

///|
pub fn TfliteRuntimeModel::model(self : TfliteRuntimeModel) -> LiteRtModel {
  self.model_
}

///|
pub fn TfliteRuntimeModel::inputs(
  self : TfliteRuntimeModel,
) -> Array[TfliteRuntimeTensor] {
  self.inputs_.copy()
}

///|
pub fn TfliteRuntimeModel::outputs(
  self : TfliteRuntimeModel,
) -> Array[TfliteRuntimeTensor] {
  self.outputs_.copy()
}

///|
fn parse_float32_buffer(
  buffer : FlatBufferTable,
  shape : @shape.Shape,
) -> Array[Float] raise LiteRtError {
  match buffer.vector(0, 1) {
    None => raise LiteRtError("TFLite float32 constant is missing buffer data")
    Some(data) => {
      if data.length != shape.element_count() * 4 {
        raise LiteRtError(
          "TFLite float32 constant has \{data.length} bytes for shape \{shape.to_string()}",
        )
      }
      Array::makei(shape.element_count(), fn(index) raise LiteRtError {
        Float::reinterpret_from_uint(data.reader.u32(data.position + index * 4))
      })
    }
  }
}

///|
fn parse_int32_buffer(
  buffer : FlatBufferTable,
  shape : @shape.Shape,
) -> Array[Int] raise LiteRtError {
  match buffer.vector(0, 1) {
    None => raise LiteRtError("TFLite int32 constant is missing buffer data")
    Some(data) => {
      if data.length != shape.element_count() * 4 {
        raise LiteRtError(
          "TFLite int32 constant has \{data.length} bytes for shape \{shape.to_string()}",
        )
      }
      Array::makei(shape.element_count(), fn(index) raise LiteRtError {
        data.reader.i32(data.position + index * 4)
      })
    }
  }
}

///|
fn normalize_quantized_dimension(
  shape : @shape.Shape,
  scales : Array[Float],
  quantized_dimension : Int,
) -> Int {
  // Some valid per-channel INT32 bias tensors retain the producer's NHWC
  // channel axis (for example 3) even though the bias itself is rank one.
  // Its scale vector still has exactly one value per bias/output channel, so
  // normalize that representation to the only axis the tensor has.
  if scales.length() > 1 &&
    shape.rank() == 1 &&
    scales.length() == shape.dimension(0) {
    0
  } else {
    quantized_dimension
  }
}

///|
fn parse_quantization(
  tensor : FlatBufferTable,
  shape : @shape.Shape,
  tensor_name : String,
) -> TfliteQuantization? raise LiteRtError {
  match tensor.table(4) {
    None => None
    Some(quantization) => {
      let scales = match quantization.vector(2, 4) {
        None => return None
        Some(vector) => vector_f32(vector)
      }
      let zero_points = vector_i64_as_int(
        required_vector(quantization, 3, 8, "quantization zero_point"),
      )
      if scales.is_empty() ||
        (zero_points.length() != 1 && zero_points.length() != scales.length()) {
        raise LiteRtError(
          "TFLite quantization scale and zero_point are incompatible for \{tensor_name}",
        )
      }
      for scale in scales {
        if scale <= 0.0 {
          raise LiteRtError(
            "TFLite quantization scales must be positive for \{tensor_name}",
          )
        }
      }
      // The current TFLite schema stores quantized_dimension at field 6.
      // Older fixture encoders used field 5 before the details union was
      // inserted, so retain that value as a compatibility fallback.
      let quantized_dimension = normalize_quantized_dimension(
        shape,
        scales,
        quantization.i32(6, quantization.i32(5, 0)),
      )
      if scales.length() > 1 &&
        (
          quantized_dimension < 0 ||
          quantized_dimension >= shape.rank() ||
          scales.length() != shape.dimension(quantized_dimension)
        ) {
        raise LiteRtError(
          "TFLite per-axis quantization is incompatible with tensor shape for \{tensor_name}: scales=\{scales.length()}, axis=\{quantized_dimension}, shape=\{shape.to_string()}",
        )
      }
      Some({ scales, zero_points, quantized_dimension })
    }
  }
}

///|
fn parse_tensor(
  tensors : FlatBufferVector,
  index : Int,
) -> TfliteTensor raise LiteRtError {
  let tensor = table_at(tensors, index, "tensor")
  let name = match tensor.string(3) {
    Some(name) if name != "" => name
    _ => tensor_name(index)
  }
  let tensor_type = tensor.u8(1, 0)
  if tensor_type != 0 &&
    tensor_type != 2 &&
    tensor_type != 3 &&
    tensor_type != 9 {
    raise LiteRtError(
      "only TFLite FLOAT32, UINT8, INT8, and control INT32 tensors are supported",
    )
  }
  let shape = @shape.Shape::new(
    vector_i32(required_vector(tensor, 0, 4, "tensor shape")),
  ) catch {
    error => raise LiteRtError(error.to_string())
  }
  let buffer_index = tensor.u32(2, UInt::default()).reinterpret_as_int()
  if buffer_index < 0 {
    raise LiteRtError("TFLite tensor buffer index is invalid")
  }
  let quantization = parse_quantization(tensor, shape, name)
  if (tensor_type == 3 || tensor_type == 9) && quantization is None {
    raise LiteRtError(
      "quantized TFLite tensor is missing quantization parameters",
    )
  }
  { name, shape, tensor_type, buffer_index, quantization }
}

///|
fn require_float32(
  tensor : TfliteTensor,
  label : String,
) -> Unit raise LiteRtError {
  if tensor.tensor_type != 0 &&
    tensor.tensor_type != 3 &&
    tensor.tensor_type != 9 {
    raise LiteRtError("TFLite \{label} must be FLOAT32, UINT8, or INT8")
  }
}

///|
fn require_float32_or_quantized_int32(
  tensor : TfliteTensor,
  label : String,
) -> Unit raise LiteRtError {
  if tensor.tensor_type == 2 {
    guard tensor.quantization is Some(_) else {
      raise LiteRtError(
        "TFLite \{label} INT32 tensor requires quantization parameters",
      )
    }
    return
  }
  require_float32(tensor, label)
}

///|
fn runtime_tensor(
  tensor : TfliteTensor,
) -> TfliteRuntimeTensor raise LiteRtError {
  require_float32(tensor, "runtime input/output")
  let data_type = match tensor.tensor_type {
    0 => Float32
    3 | 9 => {
      let quantization = match tensor.quantization {
        Some(quantization) => quantization
        None =>
          raise LiteRtError(
            "quantized TFLite runtime tensor is missing metadata",
          )
      }
      if tensor.tensor_type == 3 {
        UInt8(
          quantization.scales.copy(),
          quantization.zero_points.copy(),
          quantization.quantized_dimension,
        )
      } else {
        Int8(
          quantization.scales.copy(),
          quantization.zero_points.copy(),
          quantization.quantized_dimension,
        )
      }
    }
    _ => raise LiteRtError("TFLite runtime tensor type is unsupported")
  }
  { name_: tensor.name, shape_: tensor.shape, data_type_: data_type }
}

///|
fn require_int32(
  tensor : TfliteTensor,
  label : String,
) -> Unit raise LiteRtError {
  if tensor.tensor_type != 2 {
    raise LiteRtError("TFLite \{label} must be INT32")
  }
}

///|
fn operator_code(
  operator_codes : FlatBufferVector,
  index : Int,
) -> TfliteOperatorCode raise LiteRtError {
  let code = table_at(operator_codes, index, "operator code")
  let builtin = match code.field_position(3) {
    Some(position) => code.reader.i32(position)
    None => code.u8(0, 0)
  }
  { builtin, custom: code.string(1) }
}

///|
fn tensor_for_index(
  tensors : Array[TfliteTensor],
  index : Int,
) -> TfliteTensor raise LiteRtError {
  if index < 0 || index >= tensors.length() {
    raise LiteRtError("TFLite operator tensor index is invalid")
  }
  tensors[index]
}

///|
fn float_tensor_for_index(
  tensors : Array[TfliteTensor],
  index : Int,
  label : String,
) -> TfliteTensor raise LiteRtError {
  let tensor = tensor_for_index(tensors, index)
  require_float32(tensor, label)
  tensor
}

///|
fn constant_values(
  tensor : TfliteTensor,
  buffers : FlatBufferVector,
) -> Array[Float] raise LiteRtError {
  require_float32_or_quantized_int32(tensor, "constant")
  if tensor.buffer_index <= 0 {
    raise LiteRtError("TFLite operator requires a constant tensor")
  }
  let buffer = table_at(buffers, tensor.buffer_index, "buffer")
  match tensor.tensor_type {
    0 => parse_float32_buffer(buffer, tensor.shape)
    3 | 9 => {
      let quantization = match tensor.quantization {
        Some(quantization) => quantization
        None =>
          raise LiteRtError("quantized TFLite constant is missing quantization")
      }
      match buffer.vector(0, 1) {
        None =>
          raise LiteRtError("TFLite quantized constant is missing buffer data")
        Some(data) => {
          if data.length != tensor.shape.element_count() {
            raise LiteRtError(
              "TFLite quantized constant has an invalid buffer length",
            )
          }
          let raw_values = Array::makei(data.length, fn(index) {
            let raw = data.reader.bytes[data.position + index].to_int()
            if tensor.tensor_type == 9 && raw >= 128 {
              raw - 256
            } else {
              raw
            }
          })
          dequantize_runtime_values(
            raw_values,
            tensor.shape,
            quantization.scales,
            quantization.zero_points,
            quantization.quantized_dimension,
            if tensor.tensor_type == 3 {
              0
            } else {
              -128
            },
            if tensor.tensor_type == 3 {
              255
            } else {
              127
            },
            if tensor.tensor_type == 3 {
              "UINT8"
            } else {
              "INT8"
            },
          )
        }
      }
    }
    2 => {
      let quantization = match tensor.quantization {
        Some(quantization) => quantization
        None =>
          raise LiteRtError(
            "quantized INT32 TFLite constant is missing quantization",
          )
      }
      let raw_values = parse_int32_buffer(buffer, tensor.shape)
      Array::makei(raw_values.length(), fn(index) {
        let quantized_index = quantized_axis_index(
          tensor.shape,
          quantization.scales.length(),
          quantization.quantized_dimension,
          index,
        )
        let zero_point_index = if quantization.zero_points.length() == 1 {
          0
        } else {
          quantized_index
        }
        quantization.scales[quantized_index] *
        Float::from_int(
          raw_values[index] - quantization.zero_points[zero_point_index],
        )
      })
    }
    _ => raise LiteRtError("TFLite constant type is unsupported")
  }
}

///|
fn int32_constant_values(
  tensor : TfliteTensor,
  buffers : FlatBufferVector,
) -> Array[Int] raise LiteRtError {
  require_int32(tensor, "control constant")
  if tensor.buffer_index <= 0 {
    raise LiteRtError("TFLite operator requires an int32 constant tensor")
  }
  parse_int32_buffer(
    table_at(buffers, tensor.buffer_index, "buffer"),
    tensor.shape,
  )
}

///|
fn transpose_matrix(
  values : Array[Float],
  rows : Int,
  columns : Int,
) -> Array[Float] {
  Array::makei(values.length(), fn(index) {
    let row = index / rows
    let column = index % rows
    values[column * columns + row]
  })
}

///|
fn tflite_conv_padding(
  input : TfliteTensor,
  filter : TfliteTensor,
  output : TfliteTensor,
  stride_height : Int,
  stride_width : Int,
  dilation_height : Int,
  dilation_width : Int,
  padding : Int,
) -> Array[Int] raise LiteRtError {
  if padding == 1 {
    return [0, 0, 0, 0]
  }
  if padding != 0 {
    raise LiteRtError("unsupported TFLite CONV_2D padding")
  }
  let effective_height = (filter.shape.dimension(1) - 1) * dilation_height + 1
  let effective_width = (filter.shape.dimension(2) - 1) * dilation_width + 1
  let total_height = ((output.shape.dimension(1) - 1) * stride_height +
  effective_height -
  input.shape.dimension(1)).max(0)
  let total_width = ((output.shape.dimension(2) - 1) * stride_width +
  effective_width -
  input.shape.dimension(2)).max(0)
  [
    total_height / 2,
    total_height - total_height / 2,
    total_width / 2,
    total_width - total_width / 2,
  ]
}

///|
fn tflite_pool_padding(
  input : TfliteTensor,
  output : TfliteTensor,
  window_height : Int,
  window_width : Int,
  stride_height : Int,
  stride_width : Int,
  dilation_height : Int,
  dilation_width : Int,
  padding : Int,
) -> Array[Int] raise LiteRtError {
  if padding == 1 {
    return [0, 0, 0, 0]
  }
  if padding != 0 {
    raise LiteRtError("unsupported TFLite pool2d padding")
  }
  let effective_height = (window_height - 1) * dilation_height + 1
  let effective_width = (window_width - 1) * dilation_width + 1
  let total_height = ((output.shape.dimension(1) - 1) * stride_height +
  effective_height -
  input.shape.dimension(1)).max(0)
  let total_width = ((output.shape.dimension(2) - 1) * stride_width +
  effective_width -
  input.shape.dimension(2)).max(0)
  [
    total_height / 2,
    total_height - total_height / 2,
    total_width / 2,
    total_width - total_width / 2,
  ]
}

///|
fn filter_ohwi_to_hwio(
  values : Array[Float],
  filter : TfliteTensor,
) -> Array[Float] {
  let output_channels = filter.shape.dimension(0)
  let filter_height = filter.shape.dimension(1)
  let filter_width = filter.shape.dimension(2)
  let input_channels = filter.shape.dimension(3)
  let result = Array::make(values.length(), Float::from_int(0))
  for filter_y in 0..<filter_height {
    for filter_x in 0..<filter_width {
      for input_channel in 0..<input_channels {
        for output_channel in 0..<output_channels {
          let source = (
              (output_channel * filter_height + filter_y) * filter_width +
              filter_x
            ) *
            input_channels +
            input_channel
          let destination = (
              (filter_y * filter_width + filter_x) * input_channels +
              input_channel
            ) *
            output_channels +
            output_channel
          result[destination] = values[source]
        }
      }
    }
  }
  result
}

///|
fn fused_activation_output_name(
  activation : Int,
  output : TfliteTensor,
) -> String {
  if activation == 0 {
    output.name
  } else {
    output.name + "_fused"
  }
}

///|
/// Lower a TFLite fused activation after its primary operation's result.
fn fused_activation(
  activation : Int,
  input : String,
  output : TfliteTensor,
) -> TfliteLoweredOperator raise LiteRtError {
  let nodes = match activation {
    0 => []
    1 => [LiteRtNode::Relu(input, output.name)]
    2 => [LiteRtNode::Clamp(input, output.name, -1.0, 1.0)]
    3 => [LiteRtNode::Clamp(input, output.name, 0.0, 6.0)]
    4 => [LiteRtNode::Tanh(input, output.name)]
    _ => raise LiteRtError("unsupported TFLite fused activation: \{activation}")
  }
  { nodes, values: [] }
}

///|
fn append_fused_activation(
  activation : Int,
  unactivated_output : String,
  output : TfliteTensor,
  nodes : Array[LiteRtNode],
  values : Array[LiteRtValue],
) -> TfliteLoweredOperator raise LiteRtError {
  if activation != 0 {
    values.push(LiteRtValue::intermediate(unactivated_output, output.shape))
  }
  let fused = fused_activation(activation, unactivated_output, output)
  for node in fused.nodes {
    nodes.push(node)
  }
  { nodes, values }
}

///|
fn conv2d_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 && inputs.length() != 3 {
    raise LiteRtError("TFLite CONV_2D requires two or three inputs")
  }
  let input = tensor_for_index(tensors, inputs[0])
  let filter = tensor_for_index(tensors, inputs[1])
  require_float32(input, "CONV_2D input")
  require_float32(filter, "CONV_2D filter")
  require_float32(output, "CONV_2D output")
  if input.shape.rank() != 4 ||
    filter.shape.rank() != 4 ||
    output.shape.rank() != 4 {
    raise LiteRtError("TFLite CONV_2D currently requires rank-4 tensors")
  }
  let options = match options {
    None => raise LiteRtError("TFLite CONV_2D options are missing")
    Some(table) => table
  }
  let stride_width = options.i32(1, 1)
  let stride_height = options.i32(2, 1)
  let activation = options.u8(3, 0)
  let dilation_width = options.i32(4, 1)
  let dilation_height = options.i32(5, 1)
  if stride_width <= 0 ||
    stride_height <= 0 ||
    dilation_width <= 0 ||
    dilation_height <= 0 {
    raise LiteRtError("unsupported TFLite CONV_2D options")
  }
  let input_channels = input.shape.dimension(3)
  let filter_input_channels = filter.shape.dimension(3)
  let output_channels = filter.shape.dimension(0)
  if input_channels != filter_input_channels ||
    output.shape.dimension(0) != input.shape.dimension(0) ||
    output.shape.dimension(3) != output_channels {
    raise LiteRtError("TFLite CONV_2D shapes are incompatible")
  }
  let filter_shape = @shape.Shape::new([
    filter.shape.dimension(1),
    filter.shape.dimension(2),
    filter_input_channels,
    output_channels,
  ]) catch {
    error => raise LiteRtError(error.to_string())
  }
  let conv_options = @shape.Conv2dOptions::new_with_layout(
    tflite_conv_padding(
      input,
      filter,
      output,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      options.u8(0, 0),
    ),
    [stride_height, stride_width],
    [dilation_height, dilation_width],
    1,
    @shape.InputLayout::Nhwc,
    @shape.Conv2dFilterLayout::Hwio,
  ) catch {
    error => raise LiteRtError(error.to_string())
  }
  let converted_filter_name = filter.name + "_conv_hwio"
  let generated_values = [
    LiteRtValue::constant(
      converted_filter_name,
      filter_shape,
      filter_ohwi_to_hwio(constant_values(filter, buffers), filter),
    ),
  ]
  let unactivated_output = fused_activation_output_name(activation, output)
  if inputs.length() == 2 || inputs[2] == -1 {
    return append_fused_activation(
      activation,
      unactivated_output,
      output,
      [
        LiteRtNode::Conv2d(
          input.name,
          converted_filter_name,
          unactivated_output,
          conv_options,
        ),
      ],
      generated_values,
    )
  }
  let bias = tensor_for_index(tensors, inputs[2])
  require_float32_or_quantized_int32(bias, "CONV_2D bias")
  let bias_shape = @shape.Shape::new([output_channels]) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !bias.shape.same_as(bias_shape) {
    raise LiteRtError("TFLite CONV_2D bias shape is incompatible")
  }
  let conv_name = output.name + "_conv"
  generated_values.push(LiteRtValue::intermediate(conv_name, output.shape))
  append_fused_activation(
    activation,
    unactivated_output,
    output,
    [
      LiteRtNode::Conv2d(
        input.name,
        converted_filter_name,
        conv_name,
        conv_options,
      ),
      LiteRtNode::Add(conv_name, bias.name, unactivated_output),
    ],
    generated_values,
  )
}

///|
fn depthwise_conv2d_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 && inputs.length() != 3 {
    raise LiteRtError("TFLite DEPTHWISE_CONV_2D requires two or three inputs")
  }
  let input = tensor_for_index(tensors, inputs[0])
  let filter = tensor_for_index(tensors, inputs[1])
  require_float32(input, "DEPTHWISE_CONV_2D input")
  require_float32(filter, "DEPTHWISE_CONV_2D filter")
  require_float32(output, "DEPTHWISE_CONV_2D output")
  if input.shape.rank() != 4 ||
    filter.shape.rank() != 4 ||
    output.shape.rank() != 4 {
    raise LiteRtError(
      "TFLite DEPTHWISE_CONV_2D currently requires rank-4 tensors",
    )
  }
  let options = match options {
    None => raise LiteRtError("TFLite DEPTHWISE_CONV_2D options are missing")
    Some(table) => table
  }
  let stride_width = options.i32(1, 1)
  let stride_height = options.i32(2, 1)
  let depth_multiplier = options.i32(3, 1)
  let activation = options.u8(4, 0)
  let dilation_width = options.i32(5, 1)
  let dilation_height = options.i32(6, 1)
  if stride_width <= 0 ||
    stride_height <= 0 ||
    depth_multiplier <= 0 ||
    dilation_width <= 0 ||
    dilation_height <= 0 {
    raise LiteRtError("unsupported TFLite DEPTHWISE_CONV_2D options")
  }
  let input_channels = input.shape.dimension(3)
  let output_channels = input_channels * depth_multiplier
  if filter.shape.dimension(0) != 1 ||
    filter.shape.dimension(3) != output_channels ||
    output.shape.dimension(0) != input.shape.dimension(0) ||
    output.shape.dimension(3) != output_channels {
    raise LiteRtError("TFLite DEPTHWISE_CONV_2D shapes are incompatible")
  }
  let filter_shape = @shape.Shape::new([
    filter.shape.dimension(1),
    filter.shape.dimension(2),
    1,
    output_channels,
  ]) catch {
    error => raise LiteRtError(error.to_string())
  }
  let conv_options = @shape.Conv2dOptions::new_with_layout(
    tflite_conv_padding(
      input,
      filter,
      output,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      options.u8(0, 0),
    ),
    [stride_height, stride_width],
    [dilation_height, dilation_width],
    input_channels,
    @shape.InputLayout::Nhwc,
    @shape.Conv2dFilterLayout::Hwio,
  ) catch {
    error => raise LiteRtError(error.to_string())
  }
  let converted_filter_name = filter.name + "_depthwise_hwio"
  let generated_values = [
    LiteRtValue::constant(
      converted_filter_name,
      filter_shape,
      constant_values(filter, buffers),
    ),
  ]
  let unactivated_output = fused_activation_output_name(activation, output)
  if inputs.length() == 2 || inputs[2] == -1 {
    return append_fused_activation(
      activation,
      unactivated_output,
      output,
      [
        LiteRtNode::Conv2d(
          input.name,
          converted_filter_name,
          unactivated_output,
          conv_options,
        ),
      ],
      generated_values,
    )
  }
  let bias = tensor_for_index(tensors, inputs[2])
  require_float32_or_quantized_int32(bias, "DEPTHWISE_CONV_2D bias")
  let bias_shape = @shape.Shape::new([output_channels]) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !bias.shape.same_as(bias_shape) {
    raise LiteRtError("TFLite DEPTHWISE_CONV_2D bias shape is incompatible")
  }
  let conv_name = output.name + "_depthwise_conv"
  generated_values.push(LiteRtValue::intermediate(conv_name, output.shape))
  append_fused_activation(
    activation,
    unactivated_output,
    output,
    [
      LiteRtNode::Conv2d(
        input.name,
        converted_filter_name,
        conv_name,
        conv_options,
      ),
      LiteRtNode::Add(conv_name, bias.name, unactivated_output),
    ],
    generated_values,
  )
}

///|
fn pool2d_operator(
  maximum : Bool,
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 1 {
    raise LiteRtError("TFLite pool2d requires exactly one input")
  }
  let input = tensor_for_index(tensors, inputs[0])
  require_float32(input, "pool2d input")
  require_float32(output, "pool2d output")
  if input.shape.rank() != 4 || output.shape.rank() != 4 {
    raise LiteRtError("TFLite pool2d currently requires rank-4 tensors")
  }
  let options = match options {
    None => raise LiteRtError("TFLite pool2d options are missing")
    Some(table) => table
  }
  let stride_width = options.i32(1, 1)
  let stride_height = options.i32(2, 1)
  let window_width = options.i32(3, 1)
  let window_height = options.i32(4, 1)
  let activation = options.u8(5, 0)
  if stride_width <= 0 ||
    stride_height <= 0 ||
    window_width <= 0 ||
    window_height <= 0 {
    raise LiteRtError("unsupported TFLite pool2d options")
  }
  if input.shape.dimension(0) != output.shape.dimension(0) ||
    input.shape.dimension(3) != output.shape.dimension(3) {
    raise LiteRtError("TFLite pool2d shapes are incompatible")
  }
  let pool_options = @shape.Pool2dOptions::new_with_layout(
    [window_height, window_width],
    tflite_pool_padding(
      input,
      output,
      window_height,
      window_width,
      stride_height,
      stride_width,
      1,
      1,
      options.u8(0, 0),
    ),
    [stride_height, stride_width],
    [1, 1],
    @shape.InputLayout::Nhwc,
  ) catch {
    error => raise LiteRtError(error.to_string())
  }
  let unactivated_output = fused_activation_output_name(activation, output)
  let node = if maximum {
    LiteRtNode::MaxPool2d(input.name, unactivated_output, pool_options)
  } else {
    LiteRtNode::AveragePool2d(input.name, unactivated_output, pool_options)
  }
  append_fused_activation(activation, unactivated_output, output, [node], [])
}

///|
fn fully_connected_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 && inputs.length() != 3 {
    raise LiteRtError("TFLite FULLY_CONNECTED requires two or three inputs")
  }
  let activation = match options {
    None => 0
    Some(table) => table.u8(0, 0)
  }
  let input = tensor_for_index(tensors, inputs[0])
  let weight = tensor_for_index(tensors, inputs[1])
  require_float32(input, "FULLY_CONNECTED input")
  require_float32(weight, "FULLY_CONNECTED weight")
  require_float32(output, "FULLY_CONNECTED output")
  if input.shape.rank() < 2 ||
    weight.shape.rank() != 2 ||
    output.shape.rank() != 2 {
    raise LiteRtError(
      "TFLite FULLY_CONNECTED requires an input with rank at least 2 and rank-2 weight/output tensors",
    )
  }
  let output_channels = weight.shape.dimension(0)
  let input_channels = weight.shape.dimension(1)
  let batch_size = input.shape.dimension(0)
  let input_values_per_batch = input.shape.element_count() / batch_size
  if input_values_per_batch != input_channels ||
    output.shape.dimension(0) != input.shape.dimension(0) ||
    output.shape.dimension(1) != output_channels {
    raise LiteRtError("TFLite FULLY_CONNECTED shapes are incompatible")
  }
  let transposed_weight_shape = @shape.Shape::new([
    input_channels, output_channels,
  ]) catch {
    error => raise LiteRtError(error.to_string())
  }
  let transposed_weight_name = weight.name + "_fc_transposed"
  let generated_values = [
    LiteRtValue::constant(
      transposed_weight_name,
      transposed_weight_shape,
      transpose_matrix(
        constant_values(weight, buffers),
        output_channels,
        input_channels,
      ),
    ),
  ]
  let nodes : Array[LiteRtNode] = []
  let matmul_input = if input.shape.rank() == 2 {
    input.name
  } else {
    let flattened_name = input.name + "_fc_flattened"
    let flattened_shape = @shape.Shape::new([batch_size, input_channels]) catch {
      error => raise LiteRtError(error.to_string())
    }
    generated_values.push(
      LiteRtValue::intermediate(flattened_name, flattened_shape),
    )
    nodes.push(LiteRtNode::Reshape(input.name, flattened_name))
    flattened_name
  }
  let unactivated_output = fused_activation_output_name(activation, output)
  if inputs.length() == 2 || inputs[2] == -1 {
    nodes.push(
      LiteRtNode::Matmul(
        matmul_input, transposed_weight_name, unactivated_output,
      ),
    )
    return append_fused_activation(
      activation, unactivated_output, output, nodes, generated_values,
    )
  }
  let bias = tensor_for_index(tensors, inputs[2])
  require_float32_or_quantized_int32(bias, "FULLY_CONNECTED bias")
  let bias_shape = @shape.Shape::new([output_channels]) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !bias.shape.same_as(bias_shape) {
    raise LiteRtError("TFLite FULLY_CONNECTED bias shape is incompatible")
  }
  let matmul_name = output.name + "_fc_matmul"
  generated_values.push(LiteRtValue::intermediate(matmul_name, output.shape))
  nodes.push(
    LiteRtNode::Matmul(matmul_input, transposed_weight_name, matmul_name),
  )
  nodes.push(LiteRtNode::Add(matmul_name, bias.name, unactivated_output))
  append_fused_activation(
    activation, unactivated_output, output, nodes, generated_values,
  )
}

///|
fn validate_reshape_dimensions(
  input : TfliteTensor,
  output : TfliteTensor,
  dimensions : Array[Int],
) -> Unit raise LiteRtError {
  if dimensions.length() != output.shape.rank() {
    raise LiteRtError("TFLite RESHAPE shape rank does not match its output")
  }
  let mut inferred_dimensions = 0
  for index, dimension in dimensions {
    if dimension == -1 {
      inferred_dimensions = inferred_dimensions + 1
      continue
    }
    if dimension == 0 {
      if index >= input.shape.rank() ||
        output.shape.dimension(index) != input.shape.dimension(index) {
        raise LiteRtError("TFLite RESHAPE zero dimension is incompatible")
      }
      continue
    }
    if dimension <= 0 || output.shape.dimension(index) != dimension {
      raise LiteRtError("TFLite RESHAPE dimensions do not match its output")
    }
  }
  if inferred_dimensions > 1 {
    raise LiteRtError("TFLite RESHAPE can infer only one dimension")
  }
}

///|
fn reshape_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 1 && inputs.length() != 2 {
    raise LiteRtError(
      "TFLite RESHAPE requires one data input and an optional shape input",
    )
  }
  let input = tensor_for_index(tensors, inputs[0])
  require_float32(input, "RESHAPE input")
  require_float32(output, "RESHAPE output")
  let dimensions = if inputs.length() == 2 {
    int32_constant_values(tensor_for_index(tensors, inputs[1]), buffers)
  } else {
    match options {
      Some(table) =>
        vector_i32(required_vector(table, 0, 4, "RESHAPE new_shape"))
      None => raise LiteRtError("TFLite RESHAPE shape is missing")
    }
  }
  validate_reshape_dimensions(input, output, dimensions)
  { nodes: [LiteRtNode::Reshape(input.name, output.name)], values: [] }
}

///|
fn transpose_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 {
    raise LiteRtError("TFLite TRANSPOSE requires an input and permutation")
  }
  let input = tensor_for_index(tensors, inputs[0])
  let permutation = int32_constant_values(
    tensor_for_index(tensors, inputs[1]),
    buffers,
  )
  require_float32(input, "TRANSPOSE input")
  require_float32(output, "TRANSPOSE output")
  {
    nodes: [LiteRtNode::Transpose(input.name, output.name, permutation)],
    values: [],
  }
}

///|
fn concatenate_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() < 2 {
    raise LiteRtError("TFLite CONCATENATION requires at least two inputs")
  }
  let options = match options {
    Some(table) => table
    None => raise LiteRtError("TFLite CONCATENATION options are missing")
  }
  require_float32(output, "CONCATENATION output")
  let first = float_tensor_for_index(tensors, inputs[0], "CONCATENATION input")
  let raw_axis = options.i32(0, 0)
  let axis = if raw_axis < 0 { raw_axis + first.shape.rank() } else { raw_axis }
  first.shape.validate_axis(axis) catch {
    error => raise LiteRtError(error.to_string())
  }
  let activation = options.u8(1, 0)
  let unactivated_output = fused_activation_output_name(activation, output)
  let nodes : Array[LiteRtNode] = []
  let values : Array[LiteRtValue] = []
  let mut accumulated_name = first.name
  let mut accumulated_shape = first.shape
  for input_index in 1..<inputs.length() {
    let rhs = float_tensor_for_index(
      tensors,
      inputs[input_index],
      "CONCATENATION input",
    )
    let concatenated_shape = @shape.Shape::concat(
      accumulated_shape,
      rhs.shape,
      axis,
    ) catch {
      error => raise LiteRtError(error.to_string())
    }
    let destination = if input_index == inputs.length() - 1 {
      if !concatenated_shape.same_as(output.shape) {
        raise LiteRtError("TFLite CONCATENATION output shape is incompatible")
      }
      unactivated_output
    } else {
      let name = output.name + "_concat_" + input_index.to_string()
      values.push(LiteRtValue::intermediate(name, concatenated_shape))
      name
    }
    nodes.push(
      LiteRtNode::Concat(accumulated_name, rhs.name, destination, axis),
    )
    accumulated_name = destination
    accumulated_shape = concatenated_shape
  }
  append_fused_activation(activation, unactivated_output, output, nodes, values)
}

///|
fn mean_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 {
    raise LiteRtError("TFLite MEAN requires input and constant axes")
  }
  let input = float_tensor_for_index(tensors, inputs[0], "MEAN input")
  let raw_axes = int32_constant_values(
    tensor_for_index(tensors, inputs[1]),
    buffers,
  )
  let axes : Array[Int] = []
  let seen = Array::make(input.shape.rank(), false)
  for raw_axis in raw_axes {
    let axis = if raw_axis < 0 {
      raw_axis + input.shape.rank()
    } else {
      raw_axis
    }
    if axis < 0 || axis >= input.shape.rank() || seen[axis] {
      raise LiteRtError("TFLite MEAN axes are invalid")
    }
    seen[axis] = true
    axes.push(axis)
  }
  let keep_dimensions = output.shape.rank() == input.shape.rank()
  let expected = @shape.Shape::reduce(input.shape, axes, keep_dimensions) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !expected.same_as(output.shape) {
    raise LiteRtError("TFLite MEAN output shape is incompatible")
  }
  {
    nodes: [
      LiteRtNode::ReduceMean(input.name, output.name, axes, keep_dimensions),
    ],
    values: [],
  }
}

///|
fn gather_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
  options : FlatBufferTable?,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 2 {
    raise LiteRtError("TFLite GATHER requires input and constant indices")
  }
  let input = float_tensor_for_index(tensors, inputs[0], "GATHER input")
  let indices_tensor = tensor_for_index(tensors, inputs[1])
  let indices = int32_constant_values(indices_tensor, buffers)
  let raw_axis = match options {
    Some(table) => table.i32(0, 0)
    None => 0
  }
  let axis = if raw_axis < 0 { raw_axis + input.shape.rank() } else { raw_axis }
  let expected = @shape.Shape::gather(input.shape, indices_tensor.shape, axis) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !expected.same_as(output.shape) {
    raise LiteRtError("TFLite GATHER output shape is incompatible")
  }
  {
    nodes: [
      LiteRtNode::Gather(
        input.name,
        output.name,
        indices,
        indices_tensor.shape,
        axis,
      ),
    ],
    values: [],
  }
}

///|
fn slice_operator(
  inputs : Array[Int],
  output : TfliteTensor,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
) -> TfliteLoweredOperator raise LiteRtError {
  if inputs.length() != 3 {
    raise LiteRtError("TFLite SLICE requires input, begin, and size")
  }
  let input = float_tensor_for_index(tensors, inputs[0], "SLICE input")
  let starts = int32_constant_values(
    tensor_for_index(tensors, inputs[1]),
    buffers,
  )
  let raw_sizes = int32_constant_values(
    tensor_for_index(tensors, inputs[2]),
    buffers,
  )
  if starts.length() != input.shape.rank() ||
    raw_sizes.length() != input.shape.rank() {
    raise LiteRtError("TFLite SLICE begin and size must match input rank")
  }
  let sizes : Array[Int] = []
  for axis in 0..<input.shape.rank() {
    let size = if raw_sizes[axis] == -1 {
      input.shape.dimension(axis) - starts[axis]
    } else {
      raw_sizes[axis]
    }
    sizes.push(size)
  }
  let expected = @shape.Shape::slice(input.shape, starts, sizes) catch {
    error => raise LiteRtError(error.to_string())
  }
  if !expected.same_as(output.shape) {
    raise LiteRtError("TFLite SLICE output shape is incompatible")
  }
  {
    nodes: [LiteRtNode::Slice(input.name, output.name, starts, sizes)],
    values: [],
  }
}

///|
fn operator_nodes(
  operator : FlatBufferTable,
  operator_codes : FlatBufferVector,
  tensors : Array[TfliteTensor],
  buffers : FlatBufferVector,
) -> TfliteLoweredOperator raise LiteRtError {
  let code_index = operator.u32(0, UInt::default()).reinterpret_as_int()
  let code = operator_code(operator_codes, code_index)
  let inputs = vector_i32(required_vector(operator, 1, 4, "operator inputs"))
  let outputs = vector_i32(required_vector(operator, 2, 4, "operator outputs"))
  if outputs.length() != 1 {
    raise LiteRtError("only TFLite operators with one output are supported")
  }
  let output_tensor = tensor_for_index(tensors, outputs[0])
  require_float32(output_tensor, "operator output")
  let output = output_tensor.name
  if inputs.length() != 2 &&
    (
      code.builtin == 0 ||
      code.builtin == 18 ||
      code.builtin == 41 ||
      code.builtin == 42
    ) {
    raise LiteRtError(
      "TFLite ADD, MUL, SUB, and DIV require exactly two inputs",
    )
  }
  match code.builtin {
    0 => {
      let activation = match operator.table(4) {
        None => 0
        Some(options) => options.u8(0, 0)
      }
      let unactivated_output = fused_activation_output_name(
        activation, output_tensor,
      )
      append_fused_activation(
        activation,
        unactivated_output,
        output_tensor,
        [
          LiteRtNode::Add(
            float_tensor_for_index(tensors, inputs[0], "ADD input").name,
            float_tensor_for_index(tensors, inputs[1], "ADD input").name,
            unactivated_output,
          ),
        ],
        [],
      )
    }
    1 =>
      pool2d_operator(false, inputs, output_tensor, tensors, operator.table(4))
    2 => concatenate_operator(inputs, output_tensor, tensors, operator.table(4))
    3 =>
      conv2d_operator(
        inputs,
        tensor_for_index(tensors, outputs[0]),
        tensors,
        buffers,
        operator.table(4),
      )
    4 =>
      depthwise_conv2d_operator(
        inputs,
        output_tensor,
        tensors,
        buffers,
        operator.table(4),
      )
    9 =>
      fully_connected_operator(
        inputs,
        tensor_for_index(tensors, outputs[0]),
        tensors,
        buffers,
        operator.table(4),
      )
    14 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite LOGISTIC requires exactly one input")
      }
      {
        nodes: [
          LiteRtNode::Sigmoid(
            float_tensor_for_index(tensors, inputs[0], "LOGISTIC input").name,
            output,
          ),
        ],
        values: [],
      }
    }
    18 => {
      let activation = match operator.table(4) {
        None => 0
        Some(options) => options.u8(0, 0)
      }
      let unactivated_output = fused_activation_output_name(
        activation, output_tensor,
      )
      append_fused_activation(
        activation,
        unactivated_output,
        output_tensor,
        [
          LiteRtNode::Mul(
            float_tensor_for_index(tensors, inputs[0], "MUL input").name,
            float_tensor_for_index(tensors, inputs[1], "MUL input").name,
            unactivated_output,
          ),
        ],
        [],
      )
    }
    41 => {
      let activation = match operator.table(4) {
        None => 0
        Some(options) => options.u8(0, 0)
      }
      let unactivated_output = fused_activation_output_name(
        activation, output_tensor,
      )
      append_fused_activation(
        activation,
        unactivated_output,
        output_tensor,
        [
          LiteRtNode::Sub(
            float_tensor_for_index(tensors, inputs[0], "SUB input").name,
            float_tensor_for_index(tensors, inputs[1], "SUB input").name,
            unactivated_output,
          ),
        ],
        [],
      )
    }
    42 => {
      let activation = match operator.table(4) {
        None => 0
        Some(options) => options.u8(0, 0)
      }
      let unactivated_output = fused_activation_output_name(
        activation, output_tensor,
      )
      append_fused_activation(
        activation,
        unactivated_output,
        output_tensor,
        [
          LiteRtNode::Div(
            float_tensor_for_index(tensors, inputs[0], "DIV input").name,
            float_tensor_for_index(tensors, inputs[1], "DIV input").name,
            unactivated_output,
          ),
        ],
        [],
      )
    }
    17 =>
      pool2d_operator(true, inputs, output_tensor, tensors, operator.table(4))
    20 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite RELU_N1_TO_1 requires exactly one input")
      }
      {
        nodes: [
          LiteRtNode::Clamp(
            float_tensor_for_index(tensors, inputs[0], "RELU_N1_TO_1 input").name,
            output,
            -1.0,
            1.0,
          ),
        ],
        values: [],
      }
    }
    21 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite RELU6 requires exactly one input")
      }
      {
        nodes: [
          LiteRtNode::Clamp(
            float_tensor_for_index(tensors, inputs[0], "RELU6 input").name,
            output,
            0.0,
            6.0,
          ),
        ],
        values: [],
      }
    }
    19 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite RELU requires exactly one input")
      }
      {
        nodes: [
          LiteRtNode::Relu(
            float_tensor_for_index(tensors, inputs[0], "RELU input").name,
            output,
          ),
        ],
        values: [],
      }
    }
    25 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite SOFTMAX requires exactly one input")
      }
      let input = float_tensor_for_index(tensors, inputs[0], "SOFTMAX input")
      {
        nodes: [LiteRtNode::Softmax(input.name, output, input.shape.rank() - 1)],
        values: [],
      }
    }
    28 => {
      if inputs.length() != 1 {
        raise LiteRtError("TFLite TANH requires exactly one input")
      }
      {
        nodes: [
          LiteRtNode::Tanh(
            float_tensor_for_index(tensors, inputs[0], "TANH input").name,
            output,
          ),
        ],
        values: [],
      }
    }
    22 =>
      reshape_operator(
        inputs,
        output_tensor,
        tensors,
        buffers,
        operator.table(4),
      )
    39 => transpose_operator(inputs, output_tensor, tensors, buffers)
    40 => mean_operator(inputs, output_tensor, tensors, buffers)
    36 =>
      gather_operator(
        inputs,
        output_tensor,
        tensors,
        buffers,
        operator.table(4),
      )
    65 => slice_operator(inputs, output_tensor, tensors, buffers)
    32 =>
      match code.custom {
        Some(name) =>
          raise LiteRtError("unsupported TFLite custom operator: \{name}")
        None => raise LiteRtError("unsupported unnamed TFLite custom operator")
      }
    _ =>
      raise LiteRtError("unsupported TFLite builtin operator: \{code.builtin}")
  }
}

///|
fn parse_tflite(bytes : Bytes) -> ParsedTfliteModel raise LiteRtError {
  let reader = FlatBufferReader::new(bytes)
  let model = reader.root_table()
  let version = model.u32(0, UInt::default()).reinterpret_as_int()
  if version < 3 {
    raise LiteRtError("TFLite schema version must be at least 3")
  }
  let operator_codes = required_vector(model, 1, 4, "operator_codes")
  let subgraphs = required_vector(model, 2, 4, "subgraphs")
  if subgraphs.length != 1 {
    raise LiteRtError("only one TFLite subgraph is supported")
  }
  let buffers = required_vector(model, 4, 4, "buffers")
  let subgraph = table_at(subgraphs, 0, "subgraph")
  let tensor_tables = required_vector(subgraph, 0, 4, "tensors")
  let input_indices = vector_i32(required_vector(subgraph, 1, 4, "inputs"))
  let output_indices = vector_i32(required_vector(subgraph, 2, 4, "outputs"))
  let operators = required_vector(subgraph, 3, 4, "operators")
  // Diagnose custom operators before validating every tensor. Such models
  // commonly carry data types outside this float32 lowerer, but the custom
  // operator name is the actionable compatibility boundary.
  for index in 0..<operators.length {
    let operator = table_at(operators, index, "operator")
    let code_index = operator.u32(0, UInt::default()).reinterpret_as_int()
    let code = operator_code(operator_codes, code_index)
    if code.builtin == 32 {
      match code.custom {
        Some(name) =>
          raise LiteRtError("unsupported TFLite custom operator: \{name}")
        None => raise LiteRtError("unsupported unnamed TFLite custom operator")
      }
    }
  }
  let tensors = unique_tflite_tensor_names(
    Array::makei(tensor_tables.length, fn(index) raise LiteRtError {
      parse_tensor(tensor_tables, index)
    }),
  )
  let input_tensors = input_indices.map(fn(index) raise LiteRtError {
    let tensor = tensor_for_index(tensors, index)
    require_float32(tensor, "subgraph input")
    tensor
  })
  let input_set : Map[Int, Bool] = Map([])
  for index in input_indices {
    input_set[index] = true
  }
  let operator_output_set : Map[Int, Bool] = Map([])
  for index in 0..<operators.length {
    let operator = table_at(operators, index, "operator")
    let output_indices = vector_i32(
      required_vector(operator, 2, 4, "operator outputs"),
    )
    for output_index in output_indices {
      operator_output_set[output_index] = true
    }
  }
  let values : Array[LiteRtValue] = []
  for index, tensor in tensors {
    if tensor.tensor_type == 2 {
      if tensor.quantization is Some(_) && tensor.buffer_index > 0 {
        values.push(
          LiteRtValue::constant(
            tensor.name,
            tensor.shape,
            constant_values(tensor, buffers),
          ),
        )
      }
      continue
    }
    if input_set.contains(index) {
      values.push(LiteRtValue::input(tensor.name, tensor.shape))
    } else if operator_output_set.contains(index) {
      // Some real TFLite fixtures retain reference test data in buffers even
      // for operator outputs. Graph topology, not buffer presence, decides
      // whether a tensor is a runtime intermediate.
      values.push(LiteRtValue::intermediate(tensor.name, tensor.shape))
    } else if tensor.buffer_index == 0 {
      values.push(LiteRtValue::intermediate(tensor.name, tensor.shape))
    } else {
      values.push(
        LiteRtValue::constant(
          tensor.name,
          tensor.shape,
          constant_values(tensor, buffers),
        ),
      )
    }
  }
  let nodes : Array[LiteRtNode] = []
  for index in 0..<operators.length {
    let lowered = operator_nodes(
      table_at(operators, index, "operator"),
      operator_codes,
      tensors,
      buffers,
    )
    for value in lowered.values {
      values.push(value)
    }
    for node in lowered.nodes {
      nodes.push(node)
    }
  }
  let output_tensors = output_indices.map(fn(index) raise LiteRtError {
    let tensor = tensor_for_index(tensors, index)
    require_float32(tensor, "subgraph output")
    tensor
  })
  let outputs = output_tensors.map(fn(tensor) { tensor.name })
  {
    model: LiteRtModel::new(values, nodes, outputs),
    inputs: input_tensors,
    outputs: output_tensors,
  }
}

///|
pub fn TfliteModel::parse(bytes : Bytes) -> LiteRtModel raise LiteRtError {
  parse_tflite(bytes).model
}

///|
/// Parse a TFLite model while retaining UINT8/INT8 runtime I/O metadata.
pub fn TfliteModel::parse_runtime(
  bytes : Bytes,
) -> TfliteRuntimeModel raise LiteRtError {
  let parsed = parse_tflite(bytes)
  {
    model_: parsed.model,
    inputs_: parsed.inputs.map(runtime_tensor),
    outputs_: parsed.outputs.map(runtime_tensor),
  }
}
