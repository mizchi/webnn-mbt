///|
fn materialized_model_shape(
  dimensions : Array[Int],
) -> @shape.Shape raise @tensor.TensorError {
  @shape.Shape::new(dimensions) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
}

///|
fn WebNNGraphBuilder::materialize_attention_mask(
  self : WebNNGraphBuilder,
  mask : @model.AttentionMask?,
) -> WebNNTensor? raise @tensor.TensorError {
  match mask {
    Some(data) => Some(self.constant(data.shape(), data.values()))
    None => None
  }
}

///|
fn WebNNGraphBuilder::materialize_self_attention_with_mask(
  self : WebNNGraphBuilder,
  config : @model.SelfAttentionConfig,
  parameters : @model.SelfAttentionParameters,
  additive_mask : WebNNTensor?,
) -> @model.SelfAttention[WebNNTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "attention parameters do not match attention config",
    )
  }
  let width = config.width()
  let weight_shape = materialized_model_shape([width, width])
  let bias_shape = materialized_model_shape([width])
  {
    query: {
      weight: self.constant(weight_shape, parameters.query_weight()),
      bias: self.constant(bias_shape, parameters.query_bias()),
    },
    key: {
      weight: self.constant(weight_shape, parameters.key_weight()),
      bias: self.constant(bias_shape, parameters.key_bias()),
    },
    value: {
      weight: self.constant(weight_shape, parameters.value_weight()),
      bias: self.constant(bias_shape, parameters.value_bias()),
    },
    output: {
      weight: self.constant(weight_shape, parameters.output_weight()),
      bias: self.constant(bias_shape, parameters.output_bias()),
    },
    score_scale: self.constant(materialized_model_shape([1]), [
      config.score_scale(),
    ]),
    additive_mask,
    heads: config.heads(),
  }
}

///|
pub fn WebNNGraphBuilder::materialize_self_attention(
  self : WebNNGraphBuilder,
  config : @model.SelfAttentionConfig,
  parameters : @model.SelfAttentionParameters,
  mask : @model.AttentionMask?,
) -> @model.SelfAttention[WebNNTensor] raise @tensor.TensorError {
  self.materialize_self_attention_with_mask(
    config,
    parameters,
    self.materialize_attention_mask(mask),
  )
}

///|
pub fn WebNNGraphBuilder::materialize_feed_forward(
  self : WebNNGraphBuilder,
  config : @model.TransformerEncoderConfig,
  parameters : @model.FeedForwardParameters,
) -> @model.FeedForward[WebNNTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "feed-forward parameters do not match encoder config",
    )
  }
  let width = config.width()
  let hidden_size = config.hidden_size()
  {
    input_projection: {
      weight: self.constant(
        materialized_model_shape([width, hidden_size]),
        parameters.input_weight(),
      ),
      bias: self.constant(
        materialized_model_shape([hidden_size]),
        parameters.input_bias(),
      ),
    },
    output_projection: {
      weight: self.constant(
        materialized_model_shape([hidden_size, width]),
        parameters.output_weight(),
      ),
      bias: self.constant(
        materialized_model_shape([width]),
        parameters.output_bias(),
      ),
    },
  }
}

///|
fn WebNNGraphBuilder::materialize_transformer_encoder_with_mask(
  self : WebNNGraphBuilder,
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  additive_mask : WebNNTensor?,
) -> @model.TransformerEncoderBlock[WebNNTensor] raise @tensor.TensorError {
  let width_shape = materialized_model_shape([config.width()])
  {
    attention_normalization: {
      scale: self.constant(
        width_shape,
        parameters.attention_normalization_scale(),
      ),
      bias: self.constant(
        width_shape,
        parameters.attention_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    attention: self.materialize_self_attention_with_mask(
      config.attention(),
      parameters.attention(),
      additive_mask,
    ),
    feed_forward_normalization: {
      scale: self.constant(
        width_shape,
        parameters.feed_forward_normalization_scale(),
      ),
      bias: self.constant(
        width_shape,
        parameters.feed_forward_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    feed_forward: self.materialize_feed_forward(
      config,
      parameters.feed_forward(),
    ),
  }
}

///|
pub fn WebNNGraphBuilder::materialize_transformer_encoder(
  self : WebNNGraphBuilder,
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  mask : @model.AttentionMask?,
) -> @model.TransformerEncoderBlock[WebNNTensor] raise @tensor.TensorError {
  self.materialize_transformer_encoder_with_mask(
    config,
    parameters,
    self.materialize_attention_mask(mask),
  )
}

///|
pub fn WebNNGraphBuilder::materialize_transformer_encoder_stack(
  self : WebNNGraphBuilder,
  config : @model.TransformerEncoderStackConfig,
  parameters : @model.TransformerEncoderStackParameters,
  mask : @model.AttentionMask?,
) -> @model.TransformerEncoderStack[WebNNTensor] raise @tensor.TensorError {
  let encoder_config = config.encoder()
  let additive_mask = self.materialize_attention_mask(mask)
  let layer_parameters = parameters.layers()
  if layer_parameters.length() != config.layers() {
    raise @tensor.TensorError::new(
      "encoder stack parameters do not match stack config",
    )
  }
  let layers : Array[@model.TransformerEncoderBlock[WebNNTensor]] = []
  for parameters in layer_parameters {
    layers.push(
      self.materialize_transformer_encoder_with_mask(
        encoder_config, parameters, additive_mask,
      ),
    )
  }
  @model.TransformerEncoderStack::new(layers)
}

///|
fn WebNNGraphBuilder::materialize_bert_encoder_with_mask(
  self : WebNNGraphBuilder,
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  additive_mask : WebNNTensor?,
) -> @model.BertEncoderBlock[WebNNTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "BERT encoder parameters do not match encoder config",
    )
  }
  let width_shape = @shape.Shape::new([config.width()]) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  {
    attention: self.materialize_self_attention_with_mask(
      config.attention(),
      parameters.attention(),
      additive_mask,
    ),
    attention_output_normalization: {
      scale: self.constant(
        width_shape,
        parameters.attention_normalization_scale(),
      ),
      bias: self.constant(
        width_shape,
        parameters.attention_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    feed_forward: self.materialize_feed_forward(
      config,
      parameters.feed_forward(),
    ),
    output_normalization: {
      scale: self.constant(
        width_shape,
        parameters.feed_forward_normalization_scale(),
      ),
      bias: self.constant(
        width_shape,
        parameters.feed_forward_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
  }
}

///|
pub fn WebNNGraphBuilder::materialize_bert_encoder_stack(
  self : WebNNGraphBuilder,
  config : @model.BertEncoderConfig,
  parameters : @model.BertEncoderParameters,
  mask : @model.AttentionMask?,
) -> @model.BertEncoderStack[WebNNTensor] raise @tensor.TensorError {
  let encoder_config = config.encoder()
  let additive_mask = self.materialize_attention_mask(mask)
  let layer_parameters = parameters.layers()
  if layer_parameters.length() != config.layers() {
    raise @tensor.TensorError::new(
      "BERT encoder parameters do not match stack config",
    )
  }
  let layers : Array[@model.BertEncoderBlock[WebNNTensor]] = []
  for parameters in layer_parameters {
    layers.push(
      self.materialize_bert_encoder_with_mask(
        encoder_config, parameters, additive_mask,
      ),
    )
  }
  @model.BertEncoderStack::new(layers)
}
