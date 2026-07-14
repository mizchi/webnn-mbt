///|
priv struct FlatBufferReader {
  bytes : Bytes
}

///|
priv struct FlatBufferTable {
  reader : FlatBufferReader
  position : Int
  vtable : Int
  vtable_length : Int
  object_length : Int
}

///|
priv struct FlatBufferVector {
  reader : FlatBufferReader
  position : Int
  length : Int
  element_size : Int
}

///|
fn FlatBufferReader::new(bytes : Bytes) -> FlatBufferReader {
  { bytes, }
}

///|
fn FlatBufferReader::ensure_range(
  self : FlatBufferReader,
  offset : Int,
  length : Int,
) -> Unit raise LiteRtError {
  if offset < 0 || length < 0 || offset > self.bytes.length() - length {
    raise LiteRtError("TFLite FlatBuffer offset is outside the file")
  }
}

///|
fn FlatBufferReader::u16(
  self : FlatBufferReader,
  offset : Int,
) -> Int raise LiteRtError {
  self.ensure_range(offset, 2)
  self.bytes[offset].to_int() | (self.bytes[offset + 1].to_int() << 8)
}

///|
fn FlatBufferReader::u32(
  self : FlatBufferReader,
  offset : Int,
) -> UInt raise LiteRtError {
  self.ensure_range(offset, 4)
  self.bytes[offset].to_uint() |
  (self.bytes[offset + 1].to_uint() << 8) |
  (self.bytes[offset + 2].to_uint() << 16) |
  (self.bytes[offset + 3].to_uint() << 24)
}

///|
fn FlatBufferReader::i32(
  self : FlatBufferReader,
  offset : Int,
) -> Int raise LiteRtError {
  self.u32(offset).reinterpret_as_int()
}

///|
fn FlatBufferReader::i64_as_int(
  self : FlatBufferReader,
  offset : Int,
) -> Int raise LiteRtError {
  let low = self.i32(offset)
  let high = self.i32(offset + 4)
  if (low < 0 && high != -1) || (low >= 0 && high != 0) {
    raise LiteRtError("TFLite FlatBuffer int64 value does not fit Int")
  }
  low
}

///|
fn FlatBufferReader::offset(
  self : FlatBufferReader,
  location : Int,
) -> Int raise LiteRtError {
  let relative = self.u32(location).reinterpret_as_int()
  if relative <= 0 || location > self.bytes.length() - relative {
    raise LiteRtError("TFLite FlatBuffer indirect offset is invalid")
  }
  location + relative
}

///|
fn FlatBufferReader::table(
  self : FlatBufferReader,
  position : Int,
) -> FlatBufferTable raise LiteRtError {
  self.ensure_range(position, 4)
  let vtable_distance = self.i32(position)
  if vtable_distance == 0 {
    raise LiteRtError("TFLite FlatBuffer table has an invalid vtable offset")
  }
  let vtable = position - vtable_distance
  self.ensure_range(vtable, 4)
  let vtable_length = self.u16(vtable)
  let object_length = self.u16(vtable + 2)
  if vtable_length < 4 || vtable_length % 2 != 0 || object_length < 4 {
    raise LiteRtError("TFLite FlatBuffer table has an invalid vtable")
  }
  self.ensure_range(vtable, vtable_length)
  self.ensure_range(position, object_length)
  { reader: self, position, vtable, vtable_length, object_length }
}

///|
fn FlatBufferReader::root_table(
  self : FlatBufferReader,
) -> FlatBufferTable raise LiteRtError {
  self.ensure_range(0, 8)
  if self.bytes[4] != b'T' ||
    self.bytes[5] != b'F' ||
    self.bytes[6] != b'L' ||
    self.bytes[7] != b'3' {
    raise LiteRtError("TFLite file identifier must be TFL3")
  }
  self.table(self.offset(0))
}

///|
fn FlatBufferTable::field_position(
  self : FlatBufferTable,
  field_index : Int,
) -> Int? raise LiteRtError {
  if field_index < 0 {
    raise LiteRtError("TFLite FlatBuffer field index must be non-negative")
  }
  let entry = self.vtable + 4 + field_index * 2
  if entry + 2 > self.vtable + self.vtable_length {
    return None
  }
  let field_offset = self.reader.u16(entry)
  if field_offset == 0 {
    return None
  }
  if field_offset < 4 || field_offset >= self.object_length {
    raise LiteRtError("TFLite FlatBuffer table field offset is invalid")
  }
  Some(self.position + field_offset)
}

///|
fn FlatBufferTable::u8(
  self : FlatBufferTable,
  field_index : Int,
  default : Int,
) -> Int raise LiteRtError {
  match self.field_position(field_index) {
    None => default
    Some(position) => {
      self.reader.ensure_range(position, 1)
      self.reader.bytes[position].to_int()
    }
  }
}

///|
fn FlatBufferTable::i32(
  self : FlatBufferTable,
  field_index : Int,
  default : Int,
) -> Int raise LiteRtError {
  match self.field_position(field_index) {
    None => default
    Some(position) => self.reader.i32(position)
  }
}

///|
fn FlatBufferTable::u32(
  self : FlatBufferTable,
  field_index : Int,
  default : UInt,
) -> UInt raise LiteRtError {
  match self.field_position(field_index) {
    None => default
    Some(position) => self.reader.u32(position)
  }
}

///|
fn FlatBufferTable::table(
  self : FlatBufferTable,
  field_index : Int,
) -> FlatBufferTable? raise LiteRtError {
  match self.field_position(field_index) {
    None => None
    Some(position) => Some(self.reader.table(self.reader.offset(position)))
  }
}

///|
fn FlatBufferTable::vector(
  self : FlatBufferTable,
  field_index : Int,
  element_size : Int,
) -> FlatBufferVector? raise LiteRtError {
  if element_size <= 0 {
    raise LiteRtError("TFLite FlatBuffer vector element size must be positive")
  }
  match self.field_position(field_index) {
    None => None
    Some(position) => {
      let vector_position = self.reader.offset(position)
      self.reader.ensure_range(vector_position, 4)
      let length = self.reader.u32(vector_position).reinterpret_as_int()
      if length < 0 ||
        length >
        (self.reader.bytes.length() - vector_position - 4) / element_size {
        raise LiteRtError("TFLite FlatBuffer vector length is invalid")
      }
      Some({
        reader: self.reader,
        position: vector_position + 4,
        length,
        element_size,
      })
    }
  }
}

///|
fn FlatBufferTable::string(
  self : FlatBufferTable,
  field_index : Int,
) -> String? raise LiteRtError {
  match self.field_position(field_index) {
    None => None
    Some(position) => {
      let string_position = self.reader.offset(position)
      self.reader.ensure_range(string_position, 4)
      let byte_length = self.reader.u32(string_position).reinterpret_as_int()
      let remaining = self.reader.bytes.length() - string_position - 4
      if byte_length < 0 || byte_length >= remaining {
        raise LiteRtError("TFLite FlatBuffer string length is invalid")
      }
      let data_start = string_position + 4
      let data_end = data_start + byte_length
      if self.reader.bytes[data_end] != b'\x00' {
        raise LiteRtError("TFLite FlatBuffer string is not null terminated")
      }
      let decoded = @utf8.decode(self.reader.bytes[data_start:data_end]) catch {
        _ => raise LiteRtError("TFLite FlatBuffer string is not valid UTF-8")
      }
      Some(decoded)
    }
  }
}

///|
fn FlatBufferVector::i32(
  self : FlatBufferVector,
  index : Int,
) -> Int raise LiteRtError {
  if self.element_size != 4 || index < 0 || index >= self.length {
    raise LiteRtError("TFLite FlatBuffer int vector index is invalid")
  }
  self.reader.i32(self.position + index * 4)
}

///|
fn FlatBufferVector::table(
  self : FlatBufferVector,
  index : Int,
) -> FlatBufferTable raise LiteRtError {
  if self.element_size != 4 || index < 0 || index >= self.length {
    raise LiteRtError("TFLite FlatBuffer table vector index is invalid")
  }
  self.reader.table(self.reader.offset(self.position + index * 4))
}
