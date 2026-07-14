///|
extern "js" fn parse_safetensors_header(
  bytes : Bytes,
  header_length : Int,
) -> @core.Any =
  #| (bytes, headerLength) => {
  #|   try {
  #|     const text = new TextDecoder("utf-8", { fatal: true })
  #|       .decode(bytes.subarray(8, 8 + headerLength));
  #|     const header = JSON.parse(text);
  #|     if (header === null || Array.isArray(header) || typeof header !== "object") {
  #|       return { error: "SafeTensors header must be a JSON object", entries: [] };
  #|     }
  #|     const entries = [];
  #|     for (const [name, info] of Object.entries(header)) {
  #|       if (name === "__metadata__") continue;
  #|       if (info === null || Array.isArray(info) || typeof info !== "object") {
  #|         return { error: `SafeTensors tensor ${name} metadata must be an object`, entries: [] };
  #|       }
  #|       if (typeof info.dtype !== "string") {
  #|         return { error: `SafeTensors tensor ${name} dtype must be a string`, entries: [] };
  #|       }
  #|       if (!Array.isArray(info.shape) || !info.shape.every(Number.isSafeInteger)) {
  #|         return { error: `SafeTensors tensor ${name} shape must contain integers`, entries: [] };
  #|       }
  #|       if (!Array.isArray(info.data_offsets) || info.data_offsets.length !== 2 ||
  #|           !info.data_offsets.every((value) => Number.isSafeInteger(value) && value >= 0 && value <= 0x7fffffff)) {
  #|         return { error: `SafeTensors tensor ${name} data_offsets must contain two non-negative 32-bit integers`, entries: [] };
  #|       }
  #|       entries.push({
  #|         name,
  #|         dtype: info.dtype,
  #|         shape: info.shape,
  #|         start: info.data_offsets[0],
  #|         end: info.data_offsets[1],
  #|       });
  #|     }
  #|     entries.sort((left, right) => left.start - right.start || left.end - right.end);
  #|     return { error: "", entries };
  #|   } catch (error) {
  #|     return { error: `invalid SafeTensors header: ${error instanceof Error ? error.message : String(error)}`, entries: [] };
  #|   }
  #| }

///|
extern "js" fn safetensors_header_error(header : @core.Any) -> String =
  #| (header) => header.error

///|
extern "js" fn safetensors_header_count(header : @core.Any) -> Int =
  #| (header) => header.entries.length

///|
extern "js" fn safetensors_header_name(
  header : @core.Any,
  index : Int,
) -> String =
  #| (header, index) => header.entries[index].name

///|
extern "js" fn safetensors_header_dtype(
  header : @core.Any,
  index : Int,
) -> String =
  #| (header, index) => header.entries[index].dtype

///|
extern "js" fn safetensors_header_shape(
  header : @core.Any,
  index : Int,
) -> Array[Int] =
  #| (header, index) => header.entries[index].shape

///|
extern "js" fn safetensors_header_start(header : @core.Any, index : Int) -> Int =
  #| (header, index) => header.entries[index].start

///|
extern "js" fn safetensors_header_end(header : @core.Any, index : Int) -> Int =
  #| (header, index) => header.entries[index].end

///|
fn read_safetensors_u32(bytes : Bytes, offset : Int) -> UInt {
  for index = 0, value = UInt::default()
      index < 4
      index = index + 1, value = value |
        (bytes[offset + index].to_uint() << (index * 8)) {

  } nobreak {
    value
  }
}

///|
fn safetensors_header_length(bytes : Bytes) -> Int raise @tensor.TensorError {
  if bytes.length() < 8 {
    raise @tensor.TensorError::new(
      "SafeTensors file must contain an 8-byte header length",
    )
  }
  for index in 4..<8 {
    if bytes[index].to_int() != 0 {
      raise @tensor.TensorError::new(
        "SafeTensors header length exceeds the supported 32-bit range",
      )
    }
  }
  let length = read_safetensors_u32(bytes, 0).reinterpret_as_int()
  if length <= 0 || length > 100000000 {
    raise @tensor.TensorError::new(
      "SafeTensors header length must be between 1 and 100000000 bytes",
    )
  }
  if length > bytes.length() - 8 {
    raise @tensor.TensorError::new(
      "SafeTensors header length " +
      length.to_string() +
      " exceeds file size " +
      bytes.length().to_string(),
    )
  }
  if bytes[8].to_int() != 0x7b {
    raise @tensor.TensorError::new(
      "SafeTensors header must begin with a JSON object",
    )
  }
  length
}

///|
/// Parse a packed SafeTensors file into owned float32 named tensors.
///
/// This fixed-shape backend currently accepts only F32 tensors with positive
/// dimensions. The parser validates the complete packed data section so holes,
/// overlaps, trailing bytes, and shape/byte-count mismatches cannot be hidden.
pub fn parse_safetensors(
  bytes : Bytes,
) -> Array[NamedTensor] raise @tensor.TensorError {
  let header_length = safetensors_header_length(bytes)
  let header = parse_safetensors_header(bytes, header_length)
  let header_error = safetensors_header_error(header)
  if header_error.length() > 0 {
    raise @tensor.TensorError::new(header_error)
  }
  let data_start = 8 + header_length
  let data_length = bytes.length() - data_start
  let tensors : Array[NamedTensor] = []
  let mut packed_offset = 0
  for index in 0..<safetensors_header_count(header) {
    let name = safetensors_header_name(header, index)
    let dtype = safetensors_header_dtype(header, index)
    let start = safetensors_header_start(header, index)
    let end = safetensors_header_end(header, index)
    if dtype != "F32" {
      raise @tensor.TensorError::new(
        "SafeTensors tensor " + name + " dtype " + dtype + " is not supported",
      )
    }
    if start != packed_offset {
      raise @tensor.TensorError::new(
        "SafeTensors tensor " +
        name +
        " starts at " +
        start.to_string() +
        ", expected packed offset " +
        packed_offset.to_string(),
      )
    }
    if end < start || end > data_length {
      raise @tensor.TensorError::new(
        "SafeTensors tensor " + name + " data offsets exceed the data section",
      )
    }
    let dimensions = safetensors_header_shape(header, index)
    let shape = @shape.Shape::new(dimensions) catch {
      error =>
        raise @tensor.TensorError::new(
          "SafeTensors tensor " +
          name +
          " has unsupported shape: " +
          error.to_string(),
        )
    }
    let byte_length = end - start
    let expected_byte_length = shape.element_count() * 4
    if byte_length != expected_byte_length {
      raise @tensor.TensorError::new(
        "SafeTensors tensor " +
        name +
        " byte length " +
        byte_length.to_string() +
        " must match shape " +
        shape.to_string() +
        " byte length " +
        expected_byte_length.to_string(),
      )
    }
    let values = Array::makei(shape.element_count(), fn(value_index) {
      Float::reinterpret_from_uint(
        read_safetensors_u32(bytes, data_start + start + value_index * 4),
      )
    })
    tensors.push(NamedTensor::new(name, shape, values))
    packed_offset = end
  }
  if packed_offset != data_length {
    raise @tensor.TensorError::new(
      "SafeTensors data section has " +
      (data_length - packed_offset).to_string() +
      " trailing bytes",
    )
  }
  tensors
}
