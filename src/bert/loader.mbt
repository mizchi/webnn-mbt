///|
/// A named row-major float32 tensor from an external checkpoint.
pub struct NamedTensor {
  name_ : String
  shape_ : @shape.Shape
  values_ : Array[Float]
}

///|
pub fn NamedTensor::new(
  name : String,
  shape : @shape.Shape,
  values : Array[Float],
) -> NamedTensor raise @tensor.TensorError {
  if name.length() == 0 {
    raise @tensor.TensorError::new("named tensor name must not be empty")
  }
  if values.length() != shape.element_count() {
    raise @tensor.TensorError::new(
      "tensor " +
      name +
      " data length " +
      values.length().to_string() +
      " must match shape " +
      shape.to_string(),
    )
  }
  { name_: name, shape_: shape, values_: values.copy() }
}

///|
pub fn NamedTensor::name(self : NamedTensor) -> String {
  self.name_
}

///|
pub fn NamedTensor::shape(self : NamedTensor) -> @shape.Shape {
  self.shape_
}

///|
pub fn NamedTensor::values(self : NamedTensor) -> Array[Float] {
  self.values_.copy()
}

///|
fn require_named_tensor(
  tensors : Map[String, NamedTensor],
  name : String,
  expected_dimensions : Array[Int],
) -> NamedTensor raise @tensor.TensorError {
  let tensor = match tensors.get(name) {
    Some(tensor) => tensor
    None => raise @tensor.TensorError::new("missing BERT tensor " + name)
  }
  let actual = tensor.shape()
  let expected = @shape.Shape::new(expected_dimensions) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  if !actual.same_as(expected) {
    raise @tensor.TensorError::new(
      "BERT tensor " +
      name +
      " shape " +
      actual.to_string() +
      " must match " +
      expected.to_string(),
    )
  }
  tensor
}

///|
fn transpose_checkpoint_matrix(tensor : NamedTensor) -> Array[Float] {
  let shape = tensor.shape()
  let output_size = shape.dimension(0)
  let input_size = shape.dimension(1)
  let source = tensor.values()
  Array::makei(input_size * output_size, fn(index) {
    let input = index / output_size
    let output = index % output_size
    source[output * input_size + input]
  })
}

///|
fn load_hugging_face_layer(
  config : @model.TransformerEncoderConfig,
  tensors : Map[String, NamedTensor],
  prefix : String,
) -> @model.TransformerEncoderParameters raise @tensor.TensorError {
  let width = config.width()
  let intermediate_size = config.hidden_size()
  let query_weight = require_named_tensor(
    tensors,
    prefix + ".attention.self.query.weight",
    [width, width],
  )
  let query_bias = require_named_tensor(
    tensors,
    prefix + ".attention.self.query.bias",
    [width],
  )
  let key_weight = require_named_tensor(
    tensors,
    prefix + ".attention.self.key.weight",
    [width, width],
  )
  let key_bias = require_named_tensor(
    tensors,
    prefix + ".attention.self.key.bias",
    [width],
  )
  let value_weight = require_named_tensor(
    tensors,
    prefix + ".attention.self.value.weight",
    [width, width],
  )
  let value_bias = require_named_tensor(
    tensors,
    prefix + ".attention.self.value.bias",
    [width],
  )
  let attention_output_weight = require_named_tensor(
    tensors,
    prefix + ".attention.output.dense.weight",
    [width, width],
  )
  let attention_output_bias = require_named_tensor(
    tensors,
    prefix + ".attention.output.dense.bias",
    [width],
  )
  let attention_normalization_scale = require_named_tensor(
    tensors,
    prefix + ".attention.output.LayerNorm.weight",
    [width],
  )
  let attention_normalization_bias = require_named_tensor(
    tensors,
    prefix + ".attention.output.LayerNorm.bias",
    [width],
  )
  let intermediate_weight = require_named_tensor(
    tensors,
    prefix + ".intermediate.dense.weight",
    [intermediate_size, width],
  )
  let intermediate_bias = require_named_tensor(
    tensors,
    prefix + ".intermediate.dense.bias",
    [intermediate_size],
  )
  let output_weight = require_named_tensor(
    tensors,
    prefix + ".output.dense.weight",
    [width, intermediate_size],
  )
  let output_bias = require_named_tensor(
    tensors,
    prefix + ".output.dense.bias",
    [width],
  )
  let output_normalization_scale = require_named_tensor(
    tensors,
    prefix + ".output.LayerNorm.weight",
    [width],
  )
  let output_normalization_bias = require_named_tensor(
    tensors,
    prefix + ".output.LayerNorm.bias",
    [width],
  )
  let attention = @model.SelfAttentionParameters::new(
    config.attention(),
    transpose_checkpoint_matrix(query_weight),
    query_bias.values(),
    transpose_checkpoint_matrix(key_weight),
    key_bias.values(),
    transpose_checkpoint_matrix(value_weight),
    value_bias.values(),
    transpose_checkpoint_matrix(attention_output_weight),
    attention_output_bias.values(),
  )
  let feed_forward = @model.FeedForwardParameters::new(
    config,
    transpose_checkpoint_matrix(intermediate_weight),
    intermediate_bias.values(),
    transpose_checkpoint_matrix(output_weight),
    output_bias.values(),
  )
  @model.TransformerEncoderParameters::new(
    config,
    attention_normalization_scale.values(),
    attention_normalization_bias.values(),
    attention,
    output_normalization_scale.values(),
    output_normalization_bias.values(),
    feed_forward,
  )
}

///|
/// Load the encoder portion of a Hugging Face BERT checkpoint.
///
/// Dense weights use PyTorch `[out_features, in_features]` layout and are
/// transposed to this library's `[in_features, out_features]` Linear layout.
/// Unrelated tensors such as embeddings and pooler weights are ignored.
pub fn load_hugging_face_encoder(
  config : @model.BertEncoderConfig,
  named_tensors : Array[NamedTensor],
  encoder_prefix : String,
) -> @model.BertEncoderParameters raise @tensor.TensorError {
  if encoder_prefix.length() == 0 {
    raise @tensor.TensorError::new("BERT encoder prefix must not be empty")
  }
  let tensors : Map[String, NamedTensor] = Map([])
  for tensor in named_tensors {
    let name = tensor.name()
    if tensors.get(name) is Some(_) {
      raise @tensor.TensorError::new("duplicate BERT tensor " + name)
    }
    tensors[name] = tensor
  }
  let encoder_config = config.encoder()
  let layers : Array[@model.TransformerEncoderParameters] = []
  for layer in 0..<config.layers() {
    layers.push(
      load_hugging_face_layer(
        encoder_config,
        tensors,
        encoder_prefix + ".layer." + layer.to_string(),
      ),
    )
  }
  @model.BertEncoderParameters::new(config, layers)
}
