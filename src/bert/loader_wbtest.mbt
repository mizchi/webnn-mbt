///|
fn named(
  name : String,
  shape : Array[Int],
  values : Array[Float],
) -> NamedTensor raise @tensor.TensorError {
  let tensor_shape = @shape.Shape::new(shape) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  NamedTensor::new(name, tensor_shape, values)
}

///|
fn layer_tensors(prefix : String) -> Array[NamedTensor] raise {
  [
    named(prefix + ".attention.self.query.weight", [2, 2], [1.0, 2.0, 3.0, 4.0]),
    named(prefix + ".attention.self.query.bias", [2], [0.1, 0.2]),
    named(prefix + ".attention.self.key.weight", [2, 2], [5.0, 6.0, 7.0, 8.0]),
    named(prefix + ".attention.self.key.bias", [2], [0.3, 0.4]),
    named(prefix + ".attention.self.value.weight", [2, 2], [
      9.0, 10.0, 11.0, 12.0,
    ]),
    named(prefix + ".attention.self.value.bias", [2], [0.5, 0.6]),
    named(prefix + ".attention.output.dense.weight", [2, 2], [
      13.0, 14.0, 15.0, 16.0,
    ]),
    named(prefix + ".attention.output.dense.bias", [2], [0.7, 0.8]),
    named(prefix + ".attention.output.LayerNorm.weight", [2], [1.1, 1.2]),
    named(prefix + ".attention.output.LayerNorm.bias", [2], [0.9, 1.0]),
    named(prefix + ".intermediate.dense.weight", [3, 2], [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]),
    named(prefix + ".intermediate.dense.bias", [3], [0.1, 0.2, 0.3]),
    named(prefix + ".output.dense.weight", [2, 3], [
      7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]),
    named(prefix + ".output.dense.bias", [2], [0.4, 0.5]),
    named(prefix + ".output.LayerNorm.weight", [2], [1.3, 1.4]),
    named(prefix + ".output.LayerNorm.bias", [2], [0.6, 0.7]),
  ]
}

///|
test "Hugging Face BERT loader maps names and transposes dense matrices" {
  let config = @model.BertEncoderConfig::new(1, 2, 1, 3, 3, 1.0e-5)
  let parameters = load_hugging_face_encoder(
    config,
    layer_tensors("bert.encoder.layer.0"),
    "bert.encoder",
  )
  let layer = parameters.layers()[0]
  let attention = layer.attention()
  @test.assert_eq(attention.query_weight(), [1.0, 3.0, 2.0, 4.0])
  @test.assert_eq(attention.output_weight(), [13.0, 15.0, 14.0, 16.0])
  @test.assert_eq(layer.feed_forward().input_weight(), [
    1.0, 3.0, 5.0, 2.0, 4.0, 6.0,
  ])
  @test.assert_eq(layer.feed_forward().output_weight(), [
    7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
  ])
  @test.assert_eq(layer.attention_normalization_scale(), [1.1, 1.2])
  @test.assert_eq(layer.feed_forward_normalization_bias(), [0.6, 0.7])
}

///|
test "Hugging Face BERT loader reports a missing tensor by its full name" {
  let config = @model.BertEncoderConfig::new(1, 2, 1, 3, 3, 1.0e-5)
  let tensors = layer_tensors("bert.encoder.layer.0")
  tensors.pop() |> ignore
  try load_hugging_face_encoder(config, tensors, "bert.encoder") catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "missing BERT tensor bert.encoder.layer.0.output.LayerNorm.bias",
      )
  } noraise {
    _ => fail("expected a missing BERT tensor to be rejected")
  }
}

///|
test "Hugging Face BERT loader rejects a tensor with the wrong shape" {
  let config = @model.BertEncoderConfig::new(1, 2, 1, 3, 3, 1.0e-5)
  let tensors = layer_tensors("bert.encoder.layer.0")
  tensors[0] = named("bert.encoder.layer.0.attention.self.query.weight", [4], [
    1.0, 2.0, 3.0, 4.0,
  ])
  try load_hugging_face_encoder(config, tensors, "bert.encoder") catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "BERT tensor bert.encoder.layer.0.attention.self.query.weight shape [4] must match [2, 2]",
      )
  } noraise {
    _ => fail("expected an invalid BERT tensor shape to be rejected")
  }
}

///|
test "Hugging Face BERT loader rejects duplicate tensor names" {
  let config = @model.BertEncoderConfig::new(1, 2, 1, 3, 3, 1.0e-5)
  let tensors = layer_tensors("bert.encoder.layer.0")
  tensors.push(tensors[0])
  try load_hugging_face_encoder(config, tensors, "bert.encoder") catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "duplicate BERT tensor bert.encoder.layer.0.attention.self.query.weight",
      )
  } noraise {
    _ => fail("expected a duplicate BERT tensor to be rejected")
  }
}
