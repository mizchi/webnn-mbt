///|
fn attention_weight(width : Int, multiplier : Int) -> Array[Float] {
  Array::makei(width * width, fn(index) {
    Float::from_int((index * multiplier + multiplier) % 23 - 11) /
    Float::from_int(32)
  })
}

///|
fn attention_bias(width : Int, multiplier : Int) -> Array[Float] {
  Array::makei(width, fn(index) {
    Float::from_int((index * multiplier + 1) % 7 - 3) / Float::from_int(64)
  })
}

///|
fn attention_parameters(
  config : @model.SelfAttentionConfig,
) -> @model.SelfAttentionParameters raise @tensor.TensorError {
  attention_parameters_with_offset(config, 0)
}

///|
fn attention_parameters_with_offset(
  config : @model.SelfAttentionConfig,
  offset : Int,
) -> @model.SelfAttentionParameters raise @tensor.TensorError {
  let width = config.width()
  @model.SelfAttentionParameters::new(
    config,
    attention_weight(width, 3 + offset * 2),
    attention_bias(width, 2 + offset),
    attention_weight(width, 5 + offset * 2),
    attention_bias(width, 3 + offset),
    attention_weight(width, 7 + offset * 2),
    attention_bias(width, 5 + offset),
    attention_weight(width, 11 + offset * 2),
    attention_bias(width, 7 + offset),
  )
}

///|
fn attention_inputs(tokens : Int, width : Int) -> Array[Float] {
  Array::makei(tokens * width, fn(index) {
    Float::from_int((index * 5 + index / 3) % 31 - 15) / Float::from_int(16)
  })
}

///|
fn attention_valid_tokens(tokens : Int) -> Array[Bool] {
  if tokens == 1 {
    return [true]
  }
  let invalid_count = if tokens / 4 > 0 { tokens / 4 } else { 1 }
  Array::makei(tokens, fn(index) { index < tokens - invalid_count })
}

///|
fn attention_mask(
  mode : String,
  tokens : Int,
) -> @model.AttentionMask? raise @tensor.TensorError {
  match mode {
    "none" => None
    "causal" => Some(@model.AttentionMask::causal(tokens, -10000.0))
    "padding" =>
      Some(
        @model.AttentionMask::padding(attention_valid_tokens(tokens), -10000.0),
      )
    "causal-padding" =>
      Some(
        @model.AttentionMask::causal_padding(
          attention_valid_tokens(tokens),
          -10000.0,
        ),
      )
    _ =>
      raise @tensor.TensorError::new(
        "attention mask mode must be none, causal, padding, or causal-padding",
      )
  }
}

///|
fn attention_operations_per_sample(tokens : Int, width : Int) -> Int {
  let approximate_work = tokens * width * width * 4 +
    tokens * tokens * width * 2
  if approximate_work <= 65536 {
    5
  } else if approximate_work <= 262144 {
    3
  } else {
    1
  }
}

///|
fn attention_cpu_model(
  config : @model.SelfAttentionConfig,
  params : @model.SelfAttentionParameters,
  mask : @model.AttentionMask?,
) -> @model.SelfAttention[@cpu.CpuTensor] raise {
  @cpu.materialize_self_attention(config, params, mask)
}

///|
fn attention_webnn_model(
  builder : @webnn.WebNNGraphBuilder,
  config : @model.SelfAttentionConfig,
  params : @model.SelfAttentionParameters,
  mask : @model.AttentionMask?,
) -> @model.SelfAttention[@webnn.WebNNTensor] raise {
  builder.materialize_self_attention(config, params, mask)
}

///|
extern "js" fn make_attention_report(
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  operations_per_sample : Int,
  warmup : Int,
  iterations : Int,
  context_ms : Double,
  graph_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  cpu_samples_ms : Array[Double],
  webnn_samples_ms : Array[Double],
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (tokens, width, heads, maskMode, operationsPerSample, warmup, iterations, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnSamplesMs, maxAbsError) => ({
  #|   tokens,
  #|   width,
  #|   heads,
  #|   maskMode,
  #|   operationsPerSample,
  #|   warmup,
  #|   iterations,
  #|   contextMs,
  #|   graphMs,
  #|   compileMs,
  #|   prepareMs,
  #|   cpuSamplesMs,
  #|   webnnSamplesMs,
  #|   maxAbsError,
  #| })

///|
pub async fn self_attention(
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if tokens <= 0 || tokens > 256 {
    raise @tensor.TensorError::new(
      "attention token count must be between 1 and 256",
    )
  }
  if width <= 0 || width > 512 {
    raise @tensor.TensorError::new("attention width must be between 1 and 512")
  }
  if heads <= 0 || width % heads != 0 {
    raise @tensor.TensorError::new(
      "attention heads must be positive and divide width",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("attention warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("attention iterations must be positive")
  }
  let mask = attention_mask(mask_mode, tokens)
  let config = @model.SelfAttentionConfig::new(width, heads)
  let params = attention_parameters(config)
  let input_shape = @shape.Shape::new([tokens, width])
  let inputs = attention_inputs(tokens, width)
  let operations_per_sample = attention_operations_per_sample(tokens, width)

  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu_model = attention_cpu_model(config, params, mask)
  let mut cpu_output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      cpu_output = cpu_model.forward(cpu_input).values()
    }
  }
  let cpu_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      cpu_output = cpu_model.forward(cpu_input).values()
    }
    cpu_samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }

  let context_start = performance_now()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  let context_ms = performance_now() - context_start
  defer builder.destroy()
  let graph_start = performance_now()
  let input_binding = builder.input("input", input_shape)
  let output = attention_webnn_model(builder, config, params, mask).forward(
    input_binding.tensor(),
  )
  let graph_ms = performance_now() - graph_start
  let compile_start = performance_now()
  let session = builder.compile_single(input_binding, "output", output)
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let mut webnn_output : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      webnn_output = execution.run(inputs)
    }
  }
  let webnn_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      webnn_output = execution.run(inputs)
    }
    webnn_samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_attention_report(
    tokens,
    width,
    heads,
    mask_mode,
    operations_per_sample,
    warmup,
    iterations,
    context_ms,
    graph_ms,
    compile_ms,
    prepare_ms,
    cpu_samples_ms,
    webnn_samples_ms,
    max_abs_error(cpu_output, webnn_output),
  )
}
