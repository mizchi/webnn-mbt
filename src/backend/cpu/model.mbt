///|
fn materialized_shape(
  dimensions : Array[Int],
) -> @shape.Shape raise @tensor.TensorError {
  @shape.Shape::new(dimensions) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
}

///|
fn materialize_cpu_attention_mask(
  mask : @model.AttentionMask?,
) -> CpuTensor? raise @tensor.TensorError {
  match mask {
    Some(data) => Some(CpuTensor::new(data.shape(), data.values()))
    None => None
  }
}

///|
fn materialize_self_attention_with_mask(
  config : @model.SelfAttentionConfig,
  parameters : @model.SelfAttentionParameters,
  additive_mask : CpuTensor?,
) -> @model.SelfAttention[CpuTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "attention parameters do not match attention config",
    )
  }
  let width = config.width()
  let weight_shape = materialized_shape([width, width])
  let bias_shape = materialized_shape([width])
  {
    query: {
      weight: CpuTensor::new(weight_shape, parameters.query_weight()),
      bias: CpuTensor::new(bias_shape, parameters.query_bias()),
    },
    key: {
      weight: CpuTensor::new(weight_shape, parameters.key_weight()),
      bias: CpuTensor::new(bias_shape, parameters.key_bias()),
    },
    value: {
      weight: CpuTensor::new(weight_shape, parameters.value_weight()),
      bias: CpuTensor::new(bias_shape, parameters.value_bias()),
    },
    output: {
      weight: CpuTensor::new(weight_shape, parameters.output_weight()),
      bias: CpuTensor::new(bias_shape, parameters.output_bias()),
    },
    score_scale: CpuTensor::new(materialized_shape([1]), [config.score_scale()]),
    additive_mask,
    heads: config.heads(),
  }
}

///|
pub fn materialize_self_attention(
  config : @model.SelfAttentionConfig,
  parameters : @model.SelfAttentionParameters,
  mask : @model.AttentionMask?,
) -> @model.SelfAttention[CpuTensor] raise @tensor.TensorError {
  materialize_self_attention_with_mask(
    config,
    parameters,
    materialize_cpu_attention_mask(mask),
  )
}

///|
pub fn materialize_feed_forward(
  config : @model.TransformerEncoderConfig,
  parameters : @model.FeedForwardParameters,
) -> @model.FeedForward[CpuTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "feed-forward parameters do not match encoder config",
    )
  }
  let width = config.width()
  let hidden_size = config.hidden_size()
  {
    input_projection: {
      weight: CpuTensor::new(
        materialized_shape([width, hidden_size]),
        parameters.input_weight(),
      ),
      bias: CpuTensor::new(
        materialized_shape([hidden_size]),
        parameters.input_bias(),
      ),
    },
    output_projection: {
      weight: CpuTensor::new(
        materialized_shape([hidden_size, width]),
        parameters.output_weight(),
      ),
      bias: CpuTensor::new(
        materialized_shape([width]),
        parameters.output_bias(),
      ),
    },
  }
}

///|
fn materialize_transformer_encoder_with_mask(
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  additive_mask : CpuTensor?,
) -> @model.TransformerEncoderBlock[CpuTensor] raise @tensor.TensorError {
  let width_shape = materialized_shape([config.width()])
  {
    attention_normalization: {
      scale: CpuTensor::new(
        width_shape,
        parameters.attention_normalization_scale(),
      ),
      bias: CpuTensor::new(
        width_shape,
        parameters.attention_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    attention: materialize_self_attention_with_mask(
      config.attention(),
      parameters.attention(),
      additive_mask,
    ),
    feed_forward_normalization: {
      scale: CpuTensor::new(
        width_shape,
        parameters.feed_forward_normalization_scale(),
      ),
      bias: CpuTensor::new(
        width_shape,
        parameters.feed_forward_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    feed_forward: materialize_feed_forward(config, parameters.feed_forward()),
  }
}

///|
pub fn materialize_transformer_encoder(
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  mask : @model.AttentionMask?,
) -> @model.TransformerEncoderBlock[CpuTensor] raise @tensor.TensorError {
  materialize_transformer_encoder_with_mask(
    config,
    parameters,
    materialize_cpu_attention_mask(mask),
  )
}

///|
pub fn materialize_transformer_encoder_stack(
  config : @model.TransformerEncoderStackConfig,
  parameters : @model.TransformerEncoderStackParameters,
  mask : @model.AttentionMask?,
) -> @model.TransformerEncoderStack[CpuTensor] raise @tensor.TensorError {
  let encoder_config = config.encoder()
  let additive_mask = materialize_cpu_attention_mask(mask)
  let layer_parameters = parameters.layers()
  if layer_parameters.length() != config.layers() {
    raise @tensor.TensorError::new(
      "encoder stack parameters do not match stack config",
    )
  }
  let layers : Array[@model.TransformerEncoderBlock[CpuTensor]] = []
  for parameters in layer_parameters {
    layers.push(
      materialize_transformer_encoder_with_mask(
        encoder_config, parameters, additive_mask,
      ),
    )
  }
  @model.TransformerEncoderStack::new(layers)
}

///|
fn materialize_bert_encoder_with_mask(
  config : @model.TransformerEncoderConfig,
  parameters : @model.TransformerEncoderParameters,
  additive_mask : CpuTensor?,
) -> @model.BertEncoderBlock[CpuTensor] raise @tensor.TensorError {
  if !parameters.matches(config) {
    raise @tensor.TensorError::new(
      "BERT encoder parameters do not match encoder config",
    )
  }
  let width_shape = @shape.Shape::new([config.width()]) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  {
    attention: materialize_self_attention_with_mask(
      config.attention(),
      parameters.attention(),
      additive_mask,
    ),
    attention_output_normalization: {
      scale: CpuTensor::new(
        width_shape,
        parameters.attention_normalization_scale(),
      ),
      bias: CpuTensor::new(
        width_shape,
        parameters.attention_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
    feed_forward: materialize_feed_forward(config, parameters.feed_forward()),
    output_normalization: {
      scale: CpuTensor::new(
        width_shape,
        parameters.feed_forward_normalization_scale(),
      ),
      bias: CpuTensor::new(
        width_shape,
        parameters.feed_forward_normalization_bias(),
      ),
      axes: [config.normalization_axis()],
      epsilon: config.epsilon(),
    },
  }
}

///|
pub fn materialize_bert_encoder_stack(
  config : @model.BertEncoderConfig,
  parameters : @model.BertEncoderParameters,
  mask : @model.AttentionMask?,
) -> @model.BertEncoderStack[CpuTensor] raise @tensor.TensorError {
  let encoder_config = config.encoder()
  let additive_mask = materialize_cpu_attention_mask(mask)
  let layer_parameters = parameters.layers()
  if layer_parameters.length() != config.layers() {
    raise @tensor.TensorError::new(
      "BERT encoder parameters do not match stack config",
    )
  }
  let layers : Array[@model.BertEncoderBlock[CpuTensor]] = []
  for parameters in layer_parameters {
    layers.push(
      materialize_bert_encoder_with_mask(
        encoder_config, parameters, additive_mask,
      ),
    )
  }
  @model.BertEncoderStack::new(layers)
}
