///|
pub struct SelfAttentionConfig {
  width_ : Int
  heads_ : Int
}

///|
pub fn SelfAttentionConfig::new(
  width : Int,
  heads : Int,
) -> SelfAttentionConfig raise @tensor.TensorError {
  if width <= 0 {
    raise @tensor.TensorError::new("attention width must be positive")
  }
  if heads <= 0 {
    raise @tensor.TensorError::new("attention heads must be positive")
  }
  if width % heads != 0 {
    raise @tensor.TensorError::new(
      "attention width " +
      width.to_string() +
      " must be divisible by " +
      heads.to_string() +
      " heads",
    )
  }
  { width_: width, heads_: heads }
}

///|
pub fn SelfAttentionConfig::width(self : SelfAttentionConfig) -> Int {
  self.width_
}

///|
pub fn SelfAttentionConfig::heads(self : SelfAttentionConfig) -> Int {
  self.heads_
}

///|
pub fn SelfAttentionConfig::head_size(self : SelfAttentionConfig) -> Int {
  self.width_ / self.heads_
}

///|
pub fn SelfAttentionConfig::score_scale(self : SelfAttentionConfig) -> Float {
  Float::from_int(1) / Float::from_int(self.head_size()).sqrt()
}

///|
pub struct TransformerEncoderConfig {
  attention_ : SelfAttentionConfig
  hidden_size_ : Int
  input_rank_ : Int
  epsilon_ : Float
}

///|
pub fn TransformerEncoderConfig::new(
  width : Int,
  heads : Int,
  hidden_size : Int,
  input_rank : Int,
  epsilon : Float,
) -> TransformerEncoderConfig raise @tensor.TensorError {
  let attention = SelfAttentionConfig::new(width, heads)
  if hidden_size <= 0 {
    raise @tensor.TensorError::new("encoder hidden size must be positive")
  }
  if input_rank != 2 && input_rank != 3 {
    raise @tensor.TensorError::new("encoder input rank must be 2 or 3")
  }
  if !(epsilon > 0.0) {
    raise @tensor.TensorError::new("encoder epsilon must be positive")
  }
  {
    attention_: attention,
    hidden_size_: hidden_size,
    input_rank_: input_rank,
    epsilon_: epsilon,
  }
}

///|
pub fn TransformerEncoderConfig::attention(
  self : TransformerEncoderConfig,
) -> SelfAttentionConfig {
  self.attention_
}

///|
pub fn TransformerEncoderConfig::width(self : TransformerEncoderConfig) -> Int {
  self.attention_.width()
}

///|
pub fn TransformerEncoderConfig::heads(self : TransformerEncoderConfig) -> Int {
  self.attention_.heads()
}

///|
pub fn TransformerEncoderConfig::hidden_size(
  self : TransformerEncoderConfig,
) -> Int {
  self.hidden_size_
}

///|
pub fn TransformerEncoderConfig::input_rank(
  self : TransformerEncoderConfig,
) -> Int {
  self.input_rank_
}

///|
pub fn TransformerEncoderConfig::normalization_axis(
  self : TransformerEncoderConfig,
) -> Int {
  self.input_rank_ - 1
}

///|
pub fn TransformerEncoderConfig::epsilon(
  self : TransformerEncoderConfig,
) -> Float {
  self.epsilon_
}

///|
fn require_parameter_length(
  name : String,
  values : Array[Float],
  expected : Int,
) -> Unit raise @tensor.TensorError {
  if values.length() != expected {
    raise @tensor.TensorError::new(
      name +
      " length " +
      values.length().to_string() +
      " must match " +
      expected.to_string(),
    )
  }
}

///|
pub struct SelfAttentionParameters {
  width_ : Int
  heads_ : Int
  query_weight_ : Array[Float]
  query_bias_ : Array[Float]
  key_weight_ : Array[Float]
  key_bias_ : Array[Float]
  value_weight_ : Array[Float]
  value_bias_ : Array[Float]
  output_weight_ : Array[Float]
  output_bias_ : Array[Float]
}

///|
pub fn SelfAttentionParameters::new(
  config : SelfAttentionConfig,
  query_weight : Array[Float],
  query_bias : Array[Float],
  key_weight : Array[Float],
  key_bias : Array[Float],
  value_weight : Array[Float],
  value_bias : Array[Float],
  output_weight : Array[Float],
  output_bias : Array[Float],
) -> SelfAttentionParameters raise @tensor.TensorError {
  let width = config.width()
  let weight_size = width * width
  require_parameter_length("query weight", query_weight, weight_size)
  require_parameter_length("query bias", query_bias, width)
  require_parameter_length("key weight", key_weight, weight_size)
  require_parameter_length("key bias", key_bias, width)
  require_parameter_length("value weight", value_weight, weight_size)
  require_parameter_length("value bias", value_bias, width)
  require_parameter_length("output weight", output_weight, weight_size)
  require_parameter_length("output bias", output_bias, width)
  {
    width_: width,
    heads_: config.heads(),
    query_weight_: query_weight.copy(),
    query_bias_: query_bias.copy(),
    key_weight_: key_weight.copy(),
    key_bias_: key_bias.copy(),
    value_weight_: value_weight.copy(),
    value_bias_: value_bias.copy(),
    output_weight_: output_weight.copy(),
    output_bias_: output_bias.copy(),
  }
}

///|
pub fn SelfAttentionParameters::matches(
  self : SelfAttentionParameters,
  config : SelfAttentionConfig,
) -> Bool {
  self.width_ == config.width() && self.heads_ == config.heads()
}

///|
pub fn SelfAttentionParameters::query_weight(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.query_weight_.copy()
}

///|
pub fn SelfAttentionParameters::query_bias(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.query_bias_.copy()
}

///|
pub fn SelfAttentionParameters::key_weight(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.key_weight_.copy()
}

///|
pub fn SelfAttentionParameters::key_bias(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.key_bias_.copy()
}

///|
pub fn SelfAttentionParameters::value_weight(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.value_weight_.copy()
}

///|
pub fn SelfAttentionParameters::value_bias(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.value_bias_.copy()
}

///|
pub fn SelfAttentionParameters::output_weight(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.output_weight_.copy()
}

///|
pub fn SelfAttentionParameters::output_bias(
  self : SelfAttentionParameters,
) -> Array[Float] {
  self.output_bias_.copy()
}

///|
pub struct FeedForwardParameters {
  width_ : Int
  hidden_size_ : Int
  input_weight_ : Array[Float]
  input_bias_ : Array[Float]
  output_weight_ : Array[Float]
  output_bias_ : Array[Float]
}

///|
pub fn FeedForwardParameters::new(
  config : TransformerEncoderConfig,
  input_weight : Array[Float],
  input_bias : Array[Float],
  output_weight : Array[Float],
  output_bias : Array[Float],
) -> FeedForwardParameters raise @tensor.TensorError {
  let width = config.width()
  let hidden_size = config.hidden_size()
  require_parameter_length(
    "feed-forward input weight",
    input_weight,
    width * hidden_size,
  )
  require_parameter_length("feed-forward input bias", input_bias, hidden_size)
  require_parameter_length(
    "feed-forward output weight",
    output_weight,
    hidden_size * width,
  )
  require_parameter_length("feed-forward output bias", output_bias, width)
  {
    width_: width,
    hidden_size_: hidden_size,
    input_weight_: input_weight.copy(),
    input_bias_: input_bias.copy(),
    output_weight_: output_weight.copy(),
    output_bias_: output_bias.copy(),
  }
}

///|
pub fn FeedForwardParameters::matches(
  self : FeedForwardParameters,
  config : TransformerEncoderConfig,
) -> Bool {
  self.width_ == config.width() && self.hidden_size_ == config.hidden_size()
}

///|
pub fn FeedForwardParameters::input_weight(
  self : FeedForwardParameters,
) -> Array[Float] {
  self.input_weight_.copy()
}

///|
pub fn FeedForwardParameters::input_bias(
  self : FeedForwardParameters,
) -> Array[Float] {
  self.input_bias_.copy()
}

///|
pub fn FeedForwardParameters::output_weight(
  self : FeedForwardParameters,
) -> Array[Float] {
  self.output_weight_.copy()
}

///|
pub fn FeedForwardParameters::output_bias(
  self : FeedForwardParameters,
) -> Array[Float] {
  self.output_bias_.copy()
}

///|
pub struct TransformerEncoderParameters {
  width_ : Int
  heads_ : Int
  hidden_size_ : Int
  attention_normalization_scale_ : Array[Float]
  attention_normalization_bias_ : Array[Float]
  attention_ : SelfAttentionParameters
  feed_forward_normalization_scale_ : Array[Float]
  feed_forward_normalization_bias_ : Array[Float]
  feed_forward_ : FeedForwardParameters
}

///|
pub fn TransformerEncoderParameters::new(
  config : TransformerEncoderConfig,
  attention_normalization_scale : Array[Float],
  attention_normalization_bias : Array[Float],
  attention : SelfAttentionParameters,
  feed_forward_normalization_scale : Array[Float],
  feed_forward_normalization_bias : Array[Float],
  feed_forward : FeedForwardParameters,
) -> TransformerEncoderParameters raise @tensor.TensorError {
  let width = config.width()
  require_parameter_length(
    "attention normalization scale", attention_normalization_scale, width,
  )
  require_parameter_length(
    "attention normalization bias", attention_normalization_bias, width,
  )
  require_parameter_length(
    "feed-forward normalization scale", feed_forward_normalization_scale, width,
  )
  require_parameter_length(
    "feed-forward normalization bias", feed_forward_normalization_bias, width,
  )
  if !attention.matches(config.attention()) {
    raise @tensor.TensorError::new(
      "attention parameters do not match encoder config",
    )
  }
  if !feed_forward.matches(config) {
    raise @tensor.TensorError::new(
      "feed-forward parameters do not match encoder config",
    )
  }
  {
    width_: config.width(),
    heads_: config.heads(),
    hidden_size_: config.hidden_size(),
    attention_normalization_scale_: attention_normalization_scale.copy(),
    attention_normalization_bias_: attention_normalization_bias.copy(),
    attention_: attention,
    feed_forward_normalization_scale_: feed_forward_normalization_scale.copy(),
    feed_forward_normalization_bias_: feed_forward_normalization_bias.copy(),
    feed_forward_: feed_forward,
  }
}

///|
pub fn TransformerEncoderParameters::matches(
  self : TransformerEncoderParameters,
  config : TransformerEncoderConfig,
) -> Bool {
  self.width_ == config.width() &&
  self.heads_ == config.heads() &&
  self.hidden_size_ == config.hidden_size()
}

///|
pub fn TransformerEncoderParameters::attention_normalization_scale(
  self : TransformerEncoderParameters,
) -> Array[Float] {
  self.attention_normalization_scale_.copy()
}

///|
pub fn TransformerEncoderParameters::attention_normalization_bias(
  self : TransformerEncoderParameters,
) -> Array[Float] {
  self.attention_normalization_bias_.copy()
}

///|
pub fn TransformerEncoderParameters::attention(
  self : TransformerEncoderParameters,
) -> SelfAttentionParameters {
  self.attention_
}

///|
pub fn TransformerEncoderParameters::feed_forward_normalization_scale(
  self : TransformerEncoderParameters,
) -> Array[Float] {
  self.feed_forward_normalization_scale_.copy()
}

///|
pub fn TransformerEncoderParameters::feed_forward_normalization_bias(
  self : TransformerEncoderParameters,
) -> Array[Float] {
  self.feed_forward_normalization_bias_.copy()
}

///|
pub fn TransformerEncoderParameters::feed_forward(
  self : TransformerEncoderParameters,
) -> FeedForwardParameters {
  self.feed_forward_
}

///|
pub struct TransformerEncoderStackConfig {
  encoder_ : TransformerEncoderConfig
  layers_ : Int
}

///|
pub fn TransformerEncoderStackConfig::new(
  layers : Int,
  width : Int,
  heads : Int,
  hidden_size : Int,
  input_rank : Int,
  epsilon : Float,
) -> TransformerEncoderStackConfig raise @tensor.TensorError {
  if layers <= 0 {
    raise @tensor.TensorError::new("encoder stack layers must be positive")
  }
  {
    encoder_: TransformerEncoderConfig::new(
      width, heads, hidden_size, input_rank, epsilon,
    ),
    layers_: layers,
  }
}

///|
pub fn TransformerEncoderStackConfig::encoder(
  self : TransformerEncoderStackConfig,
) -> TransformerEncoderConfig {
  self.encoder_
}

///|
pub fn TransformerEncoderStackConfig::layers(
  self : TransformerEncoderStackConfig,
) -> Int {
  self.layers_
}

///|
pub struct TransformerEncoderStackParameters {
  layers_ : Array[TransformerEncoderParameters]
}

///|
pub fn TransformerEncoderStackParameters::new(
  config : TransformerEncoderStackConfig,
  layers : Array[TransformerEncoderParameters],
) -> TransformerEncoderStackParameters raise @tensor.TensorError {
  if layers.length() != config.layers() {
    raise @tensor.TensorError::new(
      "encoder stack parameter count " +
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
        "encoder stack layer " +
        index.to_string() +
        " parameters do not match stack config",
      )
    }
  }
  { layers_: layers.copy() }
}

///|
pub fn TransformerEncoderStackParameters::layers(
  self : TransformerEncoderStackParameters,
) -> Array[TransformerEncoderParameters] {
  self.layers_.copy()
}
