///|
/// A BERT encoder layer using the original post-normalization order.
///
/// Dropout is intentionally absent because this model represents inference.
/// SelfAttention and FeedForward remain reusable pure sublayers; this block
/// owns the residual and LayerNorm boundaries required by BERT checkpoints.
pub(all) struct BertEncoderBlock[T] {
  attention : SelfAttention[T]
  attention_output_normalization : LayerNorm[T]
  feed_forward : FeedForward[T]
  output_normalization : LayerNorm[T]
}

///|
pub fn[T : @tensor.TensorOps] BertEncoderBlock::forward(
  self : BertEncoderBlock[T],
  input : T,
) -> T raise {
  let input_shape = input.shape()
  let attention_output = self.attention.forward(input)
  require_residual_shape("BERT attention", attention_output, input_shape)
  let attention_residual = self.attention_output_normalization.forward(
    input.add(attention_output),
  )
  let feed_forward_output = self.feed_forward.forward(attention_residual)
  require_residual_shape("BERT feed-forward", feed_forward_output, input_shape)
  self.output_normalization.forward(attention_residual.add(feed_forward_output))
}

///|
/// A non-empty sequence of post-normalization BERT encoder layers.
pub struct BertEncoderStack[T] {
  layers_ : Array[BertEncoderBlock[T]]
}

///|
pub fn[T] BertEncoderStack::new(
  layers : Array[BertEncoderBlock[T]],
) -> BertEncoderStack[T] raise @tensor.TensorError {
  if layers.length() == 0 {
    raise @tensor.TensorError::new(
      "BERT encoder stack must contain at least one layer",
    )
  }
  { layers_: layers.copy() }
}

///|
pub fn[T] BertEncoderStack::layer_count(self : BertEncoderStack[T]) -> Int {
  self.layers_.length()
}

///|
pub fn[T : @tensor.TensorOps] BertEncoderStack::forward(
  self : BertEncoderStack[T],
  input : T,
) -> T raise {
  let mut output = input
  for layer in self.layers_ {
    output = layer.forward(output)
  }
  output
}

///|
/// Shape and numerical configuration for an inference-only BERT encoder.
pub struct BertEncoderConfig {
  encoder_ : TransformerEncoderConfig
  layers_ : Int
}

///|
pub fn BertEncoderConfig::new(
  layers : Int,
  width : Int,
  heads : Int,
  intermediate_size : Int,
  input_rank : Int,
  epsilon : Float,
) -> BertEncoderConfig raise @tensor.TensorError {
  if layers <= 0 {
    raise @tensor.TensorError::new("BERT encoder layers must be positive")
  }
  {
    encoder_: TransformerEncoderConfig::new(
      width, heads, intermediate_size, input_rank, epsilon,
    ),
    layers_: layers,
  }
}

///|
pub fn BertEncoderConfig::encoder(
  self : BertEncoderConfig,
) -> TransformerEncoderConfig {
  self.encoder_
}

///|
pub fn BertEncoderConfig::layers(self : BertEncoderConfig) -> Int {
  self.layers_
}

///|
pub fn BertEncoderConfig::width(self : BertEncoderConfig) -> Int {
  self.encoder_.width()
}

///|
pub fn BertEncoderConfig::heads(self : BertEncoderConfig) -> Int {
  self.encoder_.heads()
}

///|
pub fn BertEncoderConfig::intermediate_size(self : BertEncoderConfig) -> Int {
  self.encoder_.hidden_size()
}

///|
/// Owned per-layer parameters for a BERT encoder stack.
pub struct BertEncoderParameters {
  layers_ : Array[TransformerEncoderParameters]
}

///|
pub fn BertEncoderParameters::new(
  config : BertEncoderConfig,
  layers : Array[TransformerEncoderParameters],
) -> BertEncoderParameters raise @tensor.TensorError {
  if layers.length() != config.layers() {
    raise @tensor.TensorError::new(
      "BERT encoder parameter count " +
      layers.length().to_string() +
      " must match " +
      config.layers().to_string() +
      " layers",
    )
  }
  let encoder_config = config.encoder()
  for index, layer in layers {
    if !layer.matches(encoder_config) {
      raise @tensor.TensorError::new(
        "BERT encoder layer " +
        index.to_string() +
        " parameters do not match config",
      )
    }
  }
  { layers_: layers.copy() }
}

///|
pub fn BertEncoderParameters::layers(
  self : BertEncoderParameters,
) -> Array[TransformerEncoderParameters] {
  self.layers_.copy()
}
