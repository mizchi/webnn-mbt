///|
extern "js" fn make_test_safetensors(
  header : String,
  values : Array[Float],
) -> Bytes =
  #| (header, values) => {
  #|   const encoded = new TextEncoder().encode(header);
  #|   const output = new Uint8Array(8 + encoded.length + values.length * 4);
  #|   new DataView(output.buffer).setBigUint64(0, BigInt(encoded.length), true);
  #|   output.set(encoded, 8);
  #|   const view = new DataView(output.buffer);
  #|   values.forEach((value, index) => view.setFloat32(8 + encoded.length + index * 4, value, true));
  #|   return output;
  #| }

///|
test "SafeTensors parser decodes packed F32 named tensors" {
  let bytes = make_test_safetensors(
    "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}",
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  )
  let tensors = parse_safetensors(bytes)
  @test.assert_eq(tensors.length(), 2)
  @test.assert_eq(tensors[0].name(), "weight")
  @test.assert_eq(tensors[0].shape().dimensions(), [2, 2])
  @test.assert_eq(tensors[0].values(), [1.0, 2.0, 3.0, 4.0])
  @test.assert_eq(tensors[1].name(), "bias")
  @test.assert_eq(tensors[1].values(), [5.0, 6.0])
}

///|
test "SafeTensors parser rejects unsupported dtype by tensor name" {
  let bytes = make_test_safetensors(
    "{\"weight\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}}",
    [1.0],
  )
  try parse_safetensors(bytes) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "SafeTensors tensor weight dtype F16 is not supported",
      )
  } noraise {
    _ => fail("expected an unsupported SafeTensors dtype to be rejected")
  }
}

///|
test "SafeTensors parser rejects holes in the data section" {
  let bytes = make_test_safetensors(
    "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}",
    [1.0, 2.0],
  )
  try parse_safetensors(bytes) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "SafeTensors tensor weight starts at 4, expected packed offset 0",
      )
  } noraise {
    _ => fail("expected a SafeTensors data hole to be rejected")
  }
}

///|
test "SafeTensors parser rejects shape and byte length mismatch" {
  let bytes = make_test_safetensors(
    "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
    [1.0],
  )
  try parse_safetensors(bytes) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "SafeTensors tensor weight byte length 4 must match shape [2] byte length 8",
      )
  } noraise {
    _ => fail("expected a SafeTensors tensor length mismatch to be rejected")
  }
}
