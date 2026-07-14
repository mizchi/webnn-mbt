///|
test "generic Linear forward runs on the CPU backend" {
  let input = CpuTensor::new(@shape.Shape::new([1, 2]), [1.0, 2.0])
  let weight = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
  let bias = CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0])
  let linear = @model.Linear::{ weight, bias }
  @test.assert_eq(linear.forward(input).values(), [8.0, 11.0])
}

///|
test "CPU tensor computes broadcast batched matmul" {
  let lhs = CpuTensor::new(@shape.Shape::new([2, 2, 2]), [
    1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0,
  ])
  let rhs = CpuTensor::new(@shape.Shape::new([1, 2, 2]), [5.0, 6.0, 7.0, 8.0])
  let output = lhs.matmul(rhs)
  @test.assert_eq(output.shape().dimensions(), [2, 2, 2])
  @test.assert_eq(output.values(), [5.0, 6.0, 7.0, 8.0, 19.0, 22.0, 43.0, 50.0])
}

///|
test "CPU tensor broadcasts multiple batched matmul dimensions" {
  let lhs = CpuTensor::new(@shape.Shape::new([2, 1, 2, 2]), [
    1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0,
  ])
  let rhs = CpuTensor::new(@shape.Shape::new([1, 3, 2, 1]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  ])
  let output = lhs.matmul(rhs)
  @test.assert_eq(output.shape().dimensions(), [2, 3, 2, 1])
  @test.assert_eq(output.values(), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
  ])
}

///|
test "CPU tensor applies relu after broadcast addition" {
  let lhs = CpuTensor::new(@shape.Shape::new([2, 2]), [-2.0, 1.0, 2.0, -4.0])
  let rhs = CpuTensor::new(@shape.Shape::new([2]), [1.0, 2.0])
  @test.assert_eq(lhs.add(rhs).relu().values(), [0.0, 3.0, 3.0, 0.0])
}

///|
test "CPU tensor GELU follows the exact erf definition" {
  let input = CpuTensor::new(@shape.Shape::new([3]), [-1.0, 0.0, 1.0])
  let expected : Array[Float] = [-0.15865526, 0.0, 0.8413447]
  let actual = input.gelu().values()
  for index, value in actual {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-5, true)
  }
}

///|
test "CPU tensor concatenates along an inner axis" {
  let lhs = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 2.0, 3.0, 4.0])
  let rhs = CpuTensor::new(@shape.Shape::new([2, 1]), [10.0, 20.0])
  let output = lhs.concat(rhs, 1)
  @test.assert_eq(output.shape().dimensions(), [2, 3])
  @test.assert_eq(output.values(), [1.0, 2.0, 10.0, 3.0, 4.0, 20.0])
}

///|
test "CPU tensor rejects data that does not match its shape" {
  let shape = @shape.Shape::new([2, 2])
  try CpuTensor::new(shape, [1.0, 2.0]) catch {
    error =>
      @test.assert_eq(
        error.to_string(),
        "data length 2 does not match shape [2, 2]",
      )
  } noraise {
    _ => fail("expected CpuTensor::new to reject mismatched data")
  }
}

///|
test "generic MLP applies relu only between linear layers" {
  let input = CpuTensor::new(@shape.Shape::new([1, 2]), [1.0, 2.0])
  let model = @model.Mlp::{
    hidden: {
      weight: CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, -1.0, 1.0, -1.0]),
      bias: CpuTensor::new(@shape.Shape::new([2]), [0.0, 1.0]),
    },
    output: {
      weight: CpuTensor::new(@shape.Shape::new([2, 2]), [2.0, 3.0, 4.0, 5.0]),
      bias: CpuTensor::new(@shape.Shape::new([2]), [1.0, -1.0]),
    },
  }
  @test.assert_eq(model.forward(input).values(), [7.0, 8.0])
}

///|
test "generic TransformerBlock composes normalization GELU and residual" {
  let identity = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let zero_bias = CpuTensor::new(@shape.Shape::new([2]), [0.0, 0.0])
  let model = @model.TransformerBlock::{
    input_projection: { weight: identity, bias: zero_bias },
    normalization_scale: CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0]),
    normalization_bias: zero_bias,
    output_projection: { weight: identity, bias: zero_bias },
    normalization_axes: [1],
    epsilon: 1.0e-5,
  }
  let values = model
    .forward(CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 3.0, 5.0, 7.0]))
    .values()
  let expected : Array[Float] = [0.84134, 3.84134, 4.84134, 7.84134]
  for index, value in values {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-4, true)
  }
}

///|
test "generic TransformerBlock rejects a broadcast residual" {
  let feature_shape = @shape.Shape::new([2])
  let model = @model.TransformerBlock::{
    input_projection: {
      weight: CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]),
      bias: CpuTensor::new(feature_shape, [0.0, 0.0]),
    },
    normalization_scale: CpuTensor::new(feature_shape, [1.0, 1.0]),
    normalization_bias: CpuTensor::new(feature_shape, [0.0, 0.0]),
    output_projection: {
      weight: CpuTensor::new(@shape.Shape::new([2, 1]), [1.0, 1.0]),
      bias: CpuTensor::new(@shape.Shape::new([1]), [0.0]),
    },
    normalization_axes: [1],
    epsilon: 1.0e-5,
  }
  try
    model.forward(
      CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 3.0, 5.0, 7.0]),
    )
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "Transformer residual shape [2, 1] must match input shape [2, 2]",
      )
    _ => fail("expected a TensorError for the residual shape")
  } noraise {
    _ => fail("expected a broadcast residual to be rejected")
  }
}

///|
test "attention config and parameters materialize a CPU model" {
  let config = @model.SelfAttentionConfig::new(2, 2)
  let identity : Array[Float] = [1.0, 0.0, 0.0, 1.0]
  let zero_bias : Array[Float] = [0.0, 0.0]
  let params = @model.SelfAttentionParameters::new(
    config, identity, zero_bias, identity, zero_bias, identity, zero_bias, identity,
    zero_bias,
  )
  identity[0] = 9.0
  @test.assert_eq(params.query_weight()[0], 1.0)
  let exported_weight = params.query_weight()
  exported_weight[0] = 7.0
  @test.assert_eq(params.query_weight()[0], 1.0)
  let model = materialize_self_attention(config, params, None)
  let values = model
    .forward(CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]))
    .values()
  let expected : Array[Float] = [0.7310586, 0.5, 0.5, 0.7310586]
  for index, value in values {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-5, true)
  }
}

///|
test "Transformer configs reject incompatible dimensions and ranks" {
  try @model.SelfAttentionConfig::new(3, 2) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "attention width 3 must be divisible by 2 heads")
  } noraise {
    _ => fail("expected incompatible attention config to be rejected")
  }
  try @model.TransformerEncoderConfig::new(4, 2, 8, 4, 1.0e-5) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "encoder input rank must be 2 or 3")
  } noraise {
    _ => fail("expected unsupported encoder input rank to be rejected")
  }
  let config = @model.SelfAttentionConfig::new(2, 1)
  try
    @model.SelfAttentionParameters::new(
      config,
      [1.0],
      [0.0, 0.0],
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 0.0],
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 0.0],
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 0.0],
    )
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "query weight length 1 must match 4")
  } noraise {
    _ => fail("expected invalid attention parameter length to be rejected")
  }
}

///|
fn identity_self_attention(
  heads : Int,
  additive_mask : CpuTensor?,
) -> @model.SelfAttention[CpuTensor] raise {
  let identity = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let zero_bias = CpuTensor::new(@shape.Shape::new([2]), [0.0, 0.0])
  let projection = @model.Linear::{ weight: identity, bias: zero_bias }
  {
    query: projection,
    key: projection,
    value: projection,
    output: projection,
    score_scale: CpuTensor::new(@shape.Shape::new([1]), [1.0]),
    additive_mask,
    heads,
  }
}

///|
test "generic SelfAttention computes single-head attention" {
  let model = identity_self_attention(1, None)
  let values = model
    .forward(CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]))
    .values()
  let expected : Array[Float] = [0.7310586, 0.26894143, 0.26894143, 0.7310586]
  for index, value in values {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-5, true)
  }
}

///|
test "generic SelfAttention computes multi-head attention" {
  let model = identity_self_attention(2, None)
  let values = model
    .forward(CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]))
    .values()
  let expected : Array[Float] = [0.7310586, 0.5, 0.5, 0.7310586]
  for index, value in values {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-5, true)
  }
}

///|
test "generic SelfAttention computes batched multi-head attention" {
  let model = identity_self_attention(2, None)
  let output = model.forward(
    CpuTensor::new(@shape.Shape::new([2, 2, 2]), [
      1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]),
  )
  @test.assert_eq(output.shape().dimensions(), [2, 2, 2])
  let expected : Array[Float] = [
    0.7310586, 0.5, 0.5, 0.7310586, 1.0, 1.0, 1.0, 1.0,
  ]
  for index, value in output.values() {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-5, true)
  }
}

///|
test "attention masks generate causal padding and combined additive values" {
  let causal = @model.AttentionMask::causal(2, -10000.0)
  let padding = @model.AttentionMask::padding([true, false], -10000.0)
  let combined = @model.AttentionMask::causal_padding([true, false], -10000.0)
  @test.assert_eq(causal.shape().dimensions(), [1, 2, 2])
  @test.assert_eq(causal.values(), [0.0, -10000.0, 0.0, 0.0])
  @test.assert_eq(padding.values(), [0.0, -10000.0, 0.0, -10000.0])
  @test.assert_eq(combined.values(), [0.0, -10000.0, 0.0, -10000.0])
}

///|
test "attention masks support distinct padding for each batch" {
  let padding = @model.AttentionMask::batched_padding(
    [[true, true], [true, false]],
    -10000.0,
  )
  let combined = @model.AttentionMask::batched_causal_padding(
    [[true, true], [true, false]],
    -10000.0,
  )
  @test.assert_eq(padding.shape().dimensions(), [2, 1, 2, 2])
  @test.assert_eq(padding.values(), [
    0.0, 0.0, 0.0, 0.0, 0.0, -10000.0, 0.0, -10000.0,
  ])
  @test.assert_eq(combined.values(), [
    0.0, -10000.0, 0.0, 0.0, 0.0, -10000.0, 0.0, -10000.0,
  ])
}

///|
test "attention mask accepts explicit additive values" {
  let mask = @model.AttentionMask::additive(@shape.Shape::new([2, 1, 2, 2]), [
    0.0, -1.0, -2.0, -3.0, 0.0, -4.0, -5.0, -6.0,
  ])
  @test.assert_eq(mask.shape().dimensions(), [2, 1, 2, 2])
  @test.assert_eq(mask.values(), [0.0, -1.0, -2.0, -3.0, 0.0, -4.0, -5.0, -6.0])
}

///|
test "SelfAttention and AttentionMask reject invalid contracts" {
  try
    identity_self_attention(3, None).forward(
      CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]),
    )
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "self-attention width 2 must be divisible by 3 heads",
      )
    _ => fail("expected a TensorError for incompatible attention heads")
  } noraise {
    _ => fail("expected incompatible attention heads to be rejected")
  }
  try @model.AttentionMask::causal(2, 0.0) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "attention masked value must be negative")
  } noraise {
    _ => fail("expected a non-negative mask value to be rejected")
  }
  try
    @model.AttentionMask::batched_padding([[true, true], [true]], -10000.0)
  catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "attention batch 1 token count 1 must match 2")
  } noraise {
    _ => fail("expected ragged batched padding to be rejected")
  }
}

///|
test "generic SelfAttention applies causal and padding masks" {
  let input = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let causal_data = @model.AttentionMask::causal(2, -10000.0)
  let causal = identity_self_attention(
      1,
      Some(CpuTensor::new(causal_data.shape(), causal_data.values())),
    )
    .forward(input)
    .values()
  let causal_expected : Array[Float] = [1.0, 0.0, 0.26894143, 0.7310586]
  for index, value in causal {
    @test.assert_eq((value - causal_expected[index]).abs() < 1.0e-5, true)
  }
  let padding_data = @model.AttentionMask::padding([true, false], -10000.0)
  let padding = identity_self_attention(
      1,
      Some(CpuTensor::new(padding_data.shape(), padding_data.values())),
    )
    .forward(input)
    .values()
  @test.assert_eq(padding, [1.0, 0.0, 1.0, 0.0])
}

///|
fn identity_encoder_block() -> @model.TransformerEncoderBlock[CpuTensor] raise {
  let identity = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let zero_weight = CpuTensor::new(@shape.Shape::new([2, 2]), [
    0.0, 0.0, 0.0, 0.0,
  ])
  let zero_bias = CpuTensor::new(@shape.Shape::new([2]), [0.0, 0.0])
  let normalization = @model.LayerNorm::{
    scale: CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0]),
    bias: zero_bias,
    axes: [1],
    epsilon: 1.0e-5,
  }
  @model.TransformerEncoderBlock::{
    attention_normalization: normalization,
    attention: identity_self_attention(2, None),
    feed_forward_normalization: normalization,
    feed_forward: {
      input_projection: { weight: identity, bias: zero_bias },
      output_projection: { weight: zero_weight, bias: zero_bias },
    },
  }
}

///|
test "generic TransformerEncoderBlock composes attention and feed-forward" {
  let model = identity_encoder_block()
  let values = model
    .forward(CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0]))
    .values()
  let expected : Array[Float] = [1.76158, -0.76158, -0.76158, 1.76158]
  for index, value in values {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-4, true)
  }
}

///|
test "generic TransformerEncoderStack applies every layer in order" {
  let first = identity_encoder_block()
  let second = identity_encoder_block()
  let input = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let expected = second.forward(first.forward(input)).values()
  let stack = @model.TransformerEncoderStack::new([first, second])
  @test.assert_eq(stack.layer_count(), 2)
  let actual = stack.forward(input).values()
  for index, value in actual {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-6, true)
  }
}

///|
test "TransformerEncoderStack config validates layer parameter count" {
  try @model.TransformerEncoderStackConfig::new(0, 4, 2, 8, 3, 1.0e-5) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(message, "encoder stack layers must be positive")
  } noraise {
    _ => fail("expected an empty encoder stack config to be rejected")
  }
  let config = @model.TransformerEncoderStackConfig::new(2, 4, 2, 8, 3, 1.0e-5)
  try @model.TransformerEncoderStackParameters::new(config, []) catch {
    @tensor.TensorError::TensorError(message) =>
      @test.assert_eq(
        message, "encoder stack parameter count 0 must match 2 layers",
      )
  } noraise {
    _ => fail("expected missing encoder layer parameters to be rejected")
  }
}

///|
fn identity_encoder_parameters(
  config : @model.TransformerEncoderConfig,
) -> @model.TransformerEncoderParameters raise {
  let identity : Array[Float] = [1.0, 0.0, 0.0, 1.0]
  let zero_bias : Array[Float] = [0.0, 0.0]
  let attention = @model.SelfAttentionParameters::new(
    config.attention(),
    identity,
    zero_bias,
    identity,
    zero_bias,
    identity,
    zero_bias,
    identity,
    zero_bias,
  )
  let feed_forward = @model.FeedForwardParameters::new(
    config,
    identity,
    zero_bias,
    [0.0, 0.0, 0.0, 0.0],
    zero_bias,
  )
  @model.TransformerEncoderParameters::new(
    config,
    [1.0, 1.0],
    zero_bias,
    attention,
    [1.0, 1.0],
    zero_bias,
    feed_forward,
  )
}

///|
test "CPU materializer builds a TransformerEncoderStack" {
  let config = @model.TransformerEncoderStackConfig::new(2, 2, 2, 2, 2, 1.0e-5)
  let layer = identity_encoder_parameters(config.encoder())
  let params = @model.TransformerEncoderStackParameters::new(config, [
    layer, layer,
  ])
  let stack = materialize_transformer_encoder_stack(config, params, None)
  @test.assert_eq(stack.layer_count(), 2)
  let input = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let expected = identity_encoder_block()
    .forward(identity_encoder_block().forward(input))
    .values()
  let actual = stack.forward(input).values()
  for index, value in actual {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-6, true)
  }
}

///|
fn identity_bert_encoder_block() -> @model.BertEncoderBlock[CpuTensor] raise {
  let identity = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let zero_weight = CpuTensor::new(@shape.Shape::new([2, 2]), [
    0.0, 0.0, 0.0, 0.0,
  ])
  let zero_bias = CpuTensor::new(@shape.Shape::new([2]), [0.0, 0.0])
  let normalization = @model.LayerNorm::{
    scale: CpuTensor::new(@shape.Shape::new([2]), [1.0, 1.0]),
    bias: zero_bias,
    axes: [1],
    epsilon: 1.0e-5,
  }
  @model.BertEncoderBlock::{
    attention: identity_self_attention(2, None),
    attention_output_normalization: normalization,
    feed_forward: {
      input_projection: { weight: identity, bias: zero_bias },
      output_projection: { weight: zero_weight, bias: zero_bias },
    },
    output_normalization: normalization,
  }
}

///|
test "generic BertEncoderBlock uses BERT post-normalization order" {
  let model = identity_bert_encoder_block()
  let input = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let attention_residual = input.add(model.attention.forward(input))
  let attention_output = model.attention_output_normalization.forward(
    attention_residual,
  )
  let expected = model.output_normalization.forward(
    attention_output.add(model.feed_forward.forward(attention_output)),
  )
  let actual = model.forward(input)
  for index, value in actual.values() {
    @test.assert_eq((value - expected.values()[index]).abs() < 1.0e-6, true)
  }
}

///|
test "CPU materializer builds a post-normalization BertEncoderStack" {
  let config = @model.BertEncoderConfig::new(2, 2, 2, 2, 2, 1.0e-5)
  let layer = identity_encoder_parameters(config.encoder())
  let parameters = @model.BertEncoderParameters::new(config, [layer, layer])
  let stack = materialize_bert_encoder_stack(config, parameters, None)
  @test.assert_eq(stack.layer_count(), 2)
  let input = CpuTensor::new(@shape.Shape::new([2, 2]), [1.0, 0.0, 0.0, 1.0])
  let expected = identity_bert_encoder_block()
    .forward(identity_bert_encoder_block().forward(input))
    .values()
  let actual = stack.forward(input).values()
  for index, value in actual {
    @test.assert_eq((value - expected[index]).abs() < 1.0e-6, true)
  }
}

///|
test "CPU tensor computes NCHW/OIHW conv2d" {
  let input = CpuTensor::new(@shape.Shape::new([1, 1, 3, 3]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let filter = CpuTensor::new(@shape.Shape::new([1, 1, 2, 2]), [
    1.0, 0.0, 0.0, -1.0,
  ])
  let output = input.conv2d(filter, @shape.Conv2dOptions::default())
  @test.assert_eq(output.values(), [-4.0, -4.0, -4.0, -4.0])
}

///|
test "CPU tensor computes grouped conv2d" {
  let input = CpuTensor::new(@shape.Shape::new([1, 2, 2, 2]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  ])
  let filter = CpuTensor::new(@shape.Shape::new([2, 1, 1, 1]), [2.0, 3.0])
  let options = @shape.Conv2dOptions::new([0, 0, 0, 0], [1, 1], [1, 1], 2)
  @test.assert_eq(input.conv2d(filter, options).values(), [
    2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0,
  ])
}

///|
test "CPU tensor computes NHWC/HWIO conv2d" {
  let input = CpuTensor::new(@shape.Shape::new([1, 3, 3, 1]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let filter = CpuTensor::new(@shape.Shape::new([2, 2, 1, 1]), [
    1.0, 0.0, 0.0, -1.0,
  ])
  let options = @shape.Conv2dOptions::new_with_layout(
    [0, 0, 0, 0],
    [1, 1],
    [1, 1],
    1,
    @shape.InputLayout::Nhwc,
    @shape.Conv2dFilterLayout::Hwio,
  )
  @test.assert_eq(input.conv2d(filter, options).values(), [
    -4.0, -4.0, -4.0, -4.0,
  ])
}

///|
test "CPU tensor computes max and average pool2d" {
  let input = CpuTensor::new(@shape.Shape::new([1, 1, 3, 3]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let options = @shape.Pool2dOptions::new([2, 2], [0, 0, 0, 0], [1, 1], [1, 1])
  @test.assert_eq(input.max_pool2d(options).values(), [5.0, 6.0, 8.0, 9.0])
  @test.assert_eq(input.average_pool2d(options).values(), [3.0, 4.0, 6.0, 7.0])
}

///|
test "CPU tensor computes stable sigmoid and tanh activations" {
  let input = CpuTensor::new(@shape.Shape::new([3]), [-1000.0, 0.0, 1.0])
  let sigmoid = input.sigmoid().values()
  let tanh = input.tanh().values()
  @test.assert_eq(sigmoid[0], 0.0)
  @test.assert_eq(sigmoid[1], 0.5)
  @test.assert_eq((sigmoid[2] - 0.7310586).abs() < 1.0e-6, true)
  @test.assert_eq(tanh[0], -1.0)
  @test.assert_eq(tanh[1], 0.0)
  @test.assert_eq((tanh[2] - 0.7615942).abs() < 1.0e-6, true)
}

///|
test "CPU tensor clamps values for TFLite range activations" {
  let input = CpuTensor::new(@shape.Shape::new([4]), [-2.0, -0.5, 3.0, 8.0])
  @test.assert_eq(input.clamp(-1.0, 1.0).values(), [-1.0, -0.5, 1.0, 1.0])
  @test.assert_eq(input.clamp(0.0, 6.0).values(), [0.0, 0.0, 3.0, 6.0])
}

///|
test "CPU tensor computes stable softmax along an axis" {
  let input = CpuTensor::new(@shape.Shape::new([2, 3]), [
    1.0, 2.0, 3.0, 1000.0, 1001.0, 1002.0,
  ])
  let values = input.softmax(1).values()
  let expected : Array[Float] = [
    0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094,
  ]
  for index, value in values {
    let difference = value - expected[index]
    let absolute = if difference < 0.0 { -difference } else { difference }
    @test.assert_eq(absolute < 1.0e-5, true)
  }
}

///|
test "generic TinyCnn composes convolution and classifier" {
  let model = @model.TinyCnn::{
    convolution: {
      weight: CpuTensor::new(@shape.Shape::new([1, 1, 2, 2]), [
        1.0, 0.0, 0.0, 0.0,
      ]),
      bias: CpuTensor::new(@shape.Shape::new([1, 1, 1, 1]), [0.0]),
      options: @shape.Conv2dOptions::default(),
    },
    classifier: {
      weight: CpuTensor::new(@shape.Shape::new([4, 2]), [
        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
      ]),
      bias: CpuTensor::new(@shape.Shape::new([2]), [0.0, 0.0]),
    },
  }
  let input = CpuTensor::new(@shape.Shape::new([1, 1, 3, 3]), [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let values = model.forward(input).values()
  @test.assert_eq((values[0] - 0.11920292).abs() < 1.0e-5, true)
  @test.assert_eq((values[1] - 0.88079708).abs() < 1.0e-5, true)
}
