///|
/// A fixed-shape Transformer-style feed-forward block.
///
/// The first projection changes the feature width, layer normalization and
/// GELU operate in that hidden space, and the second projection restores the
/// input width before the residual addition.
pub(all) struct TransformerBlock[T] {
  input_projection : Linear[T]
  normalization_scale : T
  normalization_bias : T
  output_projection : Linear[T]
  normalization_axes : Array[Int]
  epsilon : Float
}

///|
pub fn[T : @tensor.TensorOps] TransformerBlock::forward(
  self : TransformerBlock[T],
  input : T,
) -> T raise {
  let input_shape = input.shape()
  let hidden = self.input_projection
    .forward(input)
    .layer_normalization(
      self.normalization_scale,
      self.normalization_bias,
      self.normalization_axes,
      self.epsilon,
    )
    .gelu()
  let residual = self.output_projection.forward(hidden)
  if !residual.shape().same_as(input_shape) {
    raise @tensor.TensorError::new(
      "Transformer residual shape \{residual.shape().to_string()} must match input shape \{input_shape.to_string()}",
    )
  }
  input.add(residual)
}

///|
/// Layer normalization with explicit axes and epsilon.
pub(all) struct LayerNorm[T] {
  scale : T
  bias : T
  axes : Array[Int]
  epsilon : Float
}

///|
pub fn[T : @tensor.TensorOps] LayerNorm::forward(
  self : LayerNorm[T],
  input : T,
) -> T raise {
  input.layer_normalization(self.scale, self.bias, self.axes, self.epsilon)
}

///|
/// Position-wise Transformer feed-forward network without normalization or a
/// residual connection. Those concerns belong to TransformerEncoderBlock.
pub(all) struct FeedForward[T] {
  input_projection : Linear[T]
  output_projection : Linear[T]
}

///|
pub fn[T : @tensor.TensorOps] FeedForward::forward(
  self : FeedForward[T],
  input : T,
) -> T raise {
  self.output_projection.forward(self.input_projection.forward(input).gelu())
}

///|
fn[T : @tensor.TensorOps] require_residual_shape(
  name : String,
  residual : T,
  input_shape : @shape.Shape,
) -> Unit raise @tensor.TensorError {
  if !residual.shape().same_as(input_shape) {
    raise @tensor.TensorError::new(
      "\{name} residual shape \{residual.shape().to_string()} must match input shape \{input_shape.to_string()}",
    )
  }
}

///|
/// A pre-normalization Transformer encoder block.
///
/// Both sublayers share the same backend type, so one model definition can be
/// materialized for CPU or WebNN execution. SelfAttention and FeedForward do
/// not own residual connections; this block defines both residual boundaries.
pub(all) struct TransformerEncoderBlock[T] {
  attention_normalization : LayerNorm[T]
  attention : SelfAttention[T]
  feed_forward_normalization : LayerNorm[T]
  feed_forward : FeedForward[T]
}

///|
pub fn[T : @tensor.TensorOps] TransformerEncoderBlock::forward(
  self : TransformerEncoderBlock[T],
  input : T,
) -> T raise {
  let input_shape = input.shape()
  let attention_output = self.attention.forward(
    self.attention_normalization.forward(input),
  )
  require_residual_shape("attention", attention_output, input_shape)
  let attention_residual = input.add(attention_output)
  let feed_forward_output = self.feed_forward.forward(
    self.feed_forward_normalization.forward(attention_residual),
  )
  require_residual_shape("feed-forward", feed_forward_output, input_shape)
  attention_residual.add(feed_forward_output)
}

///|
/// A non-empty sequence of pre-normalization Transformer encoder blocks.
pub struct TransformerEncoderStack[T] {
  layers_ : Array[TransformerEncoderBlock[T]]
}

///|
pub fn[T] TransformerEncoderStack::new(
  layers : Array[TransformerEncoderBlock[T]],
) -> TransformerEncoderStack[T] raise @tensor.TensorError {
  if layers.length() == 0 {
    raise @tensor.TensorError::new(
      "Transformer encoder stack must contain at least one layer",
    )
  }
  { layers_: layers.copy() }
}

///|
pub fn[T] TransformerEncoderStack::layer_count(
  self : TransformerEncoderStack[T],
) -> Int {
  self.layers_.length()
}

///|
pub fn[T : @tensor.TensorOps] TransformerEncoderStack::forward(
  self : TransformerEncoderStack[T],
  input : T,
) -> T raise {
  let mut output = input
  for layer in self.layers_ {
    output = layer.forward(output)
  }
  output
}
