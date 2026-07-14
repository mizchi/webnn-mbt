///|
/// Materializable additive attention-mask data.
///
/// The shape is `[1, tokens, tokens]` so it broadcasts across attention heads.
/// Allowed score positions contain zero and masked positions contain the
/// caller-provided negative value.
pub struct AttentionMask {
  shape_ : @shape.Shape
  values_ : Array[Float]
}

///|
pub fn AttentionMask::shape(self : AttentionMask) -> @shape.Shape {
  self.shape_
}

///|
pub fn AttentionMask::values(self : AttentionMask) -> Array[Float] {
  self.values_.copy()
}

///|
pub fn AttentionMask::additive(
  shape : @shape.Shape,
  values : Array[Float],
) -> AttentionMask raise @tensor.TensorError {
  if shape.rank() != 3 && shape.rank() != 4 {
    raise @tensor.TensorError::new(
      "additive attention mask must have rank 3 or 4",
    )
  }
  if shape.dimension(shape.rank() - 2) != shape.dimension(shape.rank() - 1) {
    raise @tensor.TensorError::new(
      "additive self-attention mask query and key dimensions must match",
    )
  }
  if values.length() != shape.element_count() {
    raise @tensor.TensorError::new(
      "attention mask data length \{values.length()} does not match shape \{shape.to_string()}",
    )
  }
  { shape_: shape, values_: values.copy() }
}

///|
fn make_attention_mask(
  valid_tokens : Array[Bool]?,
  tokens : Int,
  causal : Bool,
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  if tokens <= 0 {
    raise @tensor.TensorError::new(
      "attention mask must contain at least one token",
    )
  }
  if !(masked_value < 0.0) {
    raise @tensor.TensorError::new("attention masked value must be negative")
  }
  match valid_tokens {
    Some(valid) =>
      if valid.length() != tokens {
        raise @tensor.TensorError::new(
          "attention padding mask length must match token count",
        )
      }
    None => ()
  }
  let values = Array::make(tokens * tokens, Float::from_int(0))
  for query in 0..<tokens {
    for key in 0..<tokens {
      let padding_masked = match valid_tokens {
        Some(valid) => !valid[key]
        None => false
      }
      if padding_masked || (causal && key > query) {
        values[query * tokens + key] = masked_value
      }
    }
  }
  let shape = @shape.Shape::new([1, tokens, tokens]) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  { shape_: shape, values_: values }
}

///|
pub fn AttentionMask::causal(
  tokens : Int,
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  make_attention_mask(None, tokens, true, masked_value)
}

///|
pub fn AttentionMask::padding(
  valid_tokens : Array[Bool],
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  make_attention_mask(
    Some(valid_tokens),
    valid_tokens.length(),
    false,
    masked_value,
  )
}

///|
pub fn AttentionMask::causal_padding(
  valid_tokens : Array[Bool],
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  make_attention_mask(
    Some(valid_tokens),
    valid_tokens.length(),
    true,
    masked_value,
  )
}

///|
fn make_batched_attention_mask(
  valid_batches : Array[Array[Bool]],
  causal : Bool,
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  if valid_batches.length() == 0 {
    raise @tensor.TensorError::new(
      "batched attention mask must contain at least one batch",
    )
  }
  if !(masked_value < 0.0) {
    raise @tensor.TensorError::new("attention masked value must be negative")
  }
  let tokens = valid_batches[0].length()
  if tokens == 0 {
    raise @tensor.TensorError::new(
      "attention mask must contain at least one token",
    )
  }
  for batch, valid_tokens in valid_batches {
    if valid_tokens.length() != tokens {
      raise @tensor.TensorError::new(
        "attention batch \{batch} token count \{valid_tokens.length()} must match \{tokens}",
      )
    }
  }
  let values = Array::make(
    valid_batches.length() * tokens * tokens,
    Float::from_int(0),
  )
  for batch, valid_tokens in valid_batches {
    let batch_offset = batch * tokens * tokens
    for query in 0..<tokens {
      for key in 0..<tokens {
        if !valid_tokens[key] || (causal && key > query) {
          values[batch_offset + query * tokens + key] = masked_value
        }
      }
    }
  }
  let shape = @shape.Shape::new([valid_batches.length(), 1, tokens, tokens]) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  { shape_: shape, values_: values }
}

///|
pub fn AttentionMask::batched_padding(
  valid_batches : Array[Array[Bool]],
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  make_batched_attention_mask(valid_batches, false, masked_value)
}

///|
pub fn AttentionMask::batched_causal_padding(
  valid_batches : Array[Array[Bool]],
  masked_value : Float,
) -> AttentionMask raise @tensor.TensorError {
  make_batched_attention_mask(valid_batches, true, masked_value)
}

///|
/// Fixed-shape self-attention shared by CPU and WebNN backends.
///
/// Input and output use `[tokens, width]` or `[batch, tokens, width]`.
/// Internally, projections are reshaped by head; batched matmul computes every
/// batch and head together. This layer returns only the projected attention;
/// residual addition belongs to a Transformer block.
pub(all) struct SelfAttention[T] {
  query : Linear[T]
  key : Linear[T]
  value : Linear[T]
  output : Linear[T]
  score_scale : T
  additive_mask : T?
  heads : Int
}

///|
fn[T : @tensor.TensorOps] require_attention_shape(
  name : String,
  tensor : T,
  expected : @shape.Shape,
) -> Unit raise @tensor.TensorError {
  if !tensor.shape().same_as(expected) {
    raise @tensor.TensorError::new(
      "attention \{name} shape \{tensor.shape().to_string()} must match \{expected.to_string()}",
    )
  }
}

///|
pub fn[T : @tensor.TensorOps] SelfAttention::forward(
  self : SelfAttention[T],
  input : T,
) -> T raise {
  let input_shape = input.shape()
  let rank = input_shape.rank()
  if rank != 2 && rank != 3 {
    raise @tensor.TensorError::new(
      "self-attention input must have shape [tokens, width] or [batch, tokens, width]",
    )
  }
  if self.heads <= 0 {
    raise @tensor.TensorError::new("self-attention heads must be positive")
  }
  let tokens = input_shape.dimension(rank - 2)
  let width = input_shape.dimension(rank - 1)
  if width % self.heads != 0 {
    raise @tensor.TensorError::new(
      "self-attention width \{width} must be divisible by \{self.heads} heads",
    )
  }
  let query = self.query.forward(input)
  let key = self.key.forward(input)
  let value = self.value.forward(input)
  require_attention_shape("query projection", query, input_shape)
  require_attention_shape("key projection", key, input_shape)
  require_attention_shape("value projection", value, input_shape)

  let head_size = width / self.heads
  let split_shape = if rank == 2 {
    @shape.Shape::new([tokens, self.heads, head_size])
  } else {
    @shape.Shape::new([input_shape.dimension(0), tokens, self.heads, head_size])
  }
  let query_permutation = if rank == 2 { [1, 0, 2] } else { [0, 2, 1, 3] }
  let key_permutation = if rank == 2 { [1, 2, 0] } else { [0, 2, 3, 1] }
  let query_heads = query.reshape(split_shape).transpose(query_permutation)
  let key_heads = key.reshape(split_shape).transpose(key_permutation)
  let value_heads = value.reshape(split_shape).transpose(query_permutation)
  let score_shape = if rank == 2 {
    @shape.Shape::new([self.heads, tokens, tokens])
  } else {
    @shape.Shape::new([input_shape.dimension(0), self.heads, tokens, tokens])
  }
  let scaled_scores = query_heads.matmul(key_heads).mul(self.score_scale)
  require_attention_shape("scaled score", scaled_scores, score_shape)
  let masked_scores = match self.additive_mask {
    Some(mask) => {
      let result = scaled_scores.add(mask)
      require_attention_shape("masked score", result, score_shape)
      result
    }
    None => scaled_scores
  }
  let attention = masked_scores.softmax(rank)
  let context = attention
    .matmul(value_heads)
    .transpose(query_permutation)
    .reshape(input_shape)
  let projected = self.output.forward(context)
  require_attention_shape("output projection", projected, input_shape)
  projected
}
