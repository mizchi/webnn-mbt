///|
fn encoder_inputs(batch_size : Int, tokens : Int, width : Int) -> Array[Float] {
  Array::makei(batch_size * tokens * width, fn(index) {
    Float::from_int(
      (index * 7 + index / 5 + index / (tokens * width)) % 37 - 18,
    ) /
    Float::from_int(16)
  })
}

///|
fn encoder_operations_per_sample(
  batch_size : Int,
  tokens : Int,
  width : Int,
) -> Int {
  let approximate_work = batch_size *
    (tokens * width * width * 8 + tokens * tokens * width * 2)
  if approximate_work <= 65536 {
    5
  } else if approximate_work <= 262144 {
    3
  } else {
    1
  }
}

///|
fn encoder_normalization_scale(width : Int, offset : Int) -> Array[Float] {
  Array::makei(width, fn(index) {
    Float::from_int(14 + (index + offset) % 5) / Float::from_int(16)
  })
}

///|
fn encoder_normalization_bias(width : Int, offset : Int) -> Array[Float] {
  Array::makei(width, fn(index) {
    Float::from_int((index + offset) % 5 - 2) / Float::from_int(128)
  })
}

///|
fn encoder_valid_batches(batch_size : Int, tokens : Int) -> Array[Array[Bool]] {
  Array::makei(batch_size, fn(batch) {
    let invalid_tokens = batch % tokens
    Array::makei(tokens, fn(token) { token < tokens - invalid_tokens })
  })
}

///|
fn encoder_attention_mask(
  mode : String,
  batch_size : Int,
  tokens : Int,
) -> @model.AttentionMask? raise @tensor.TensorError {
  match mode {
    "none" => None
    "causal" => Some(@model.AttentionMask::causal(tokens, -10000.0))
    "padding" =>
      Some(
        @model.AttentionMask::batched_padding(
          encoder_valid_batches(batch_size, tokens),
          -10000.0,
        ),
      )
    "causal-padding" =>
      Some(
        @model.AttentionMask::batched_causal_padding(
          encoder_valid_batches(batch_size, tokens),
          -10000.0,
        ),
      )
    _ =>
      raise @tensor.TensorError::new(
        "encoder mask mode must be none, causal, padding, or causal-padding",
      )
  }
}

///|
fn encoder_parameters(
  config : @model.TransformerEncoderConfig,
  layer : Int,
) -> @model.TransformerEncoderParameters raise @tensor.TensorError {
  @model.TransformerEncoderParameters::new(
    config,
    encoder_normalization_scale(config.width(), layer * 2),
    encoder_normalization_bias(config.width(), layer * 2),
    attention_parameters_with_offset(config.attention(), layer),
    encoder_normalization_scale(config.width(), layer * 2 + 1),
    encoder_normalization_bias(config.width(), layer * 2 + 1),
    transformer_feed_forward_parameters_with_offset(config, layer),
  )
}

///|
extern "js" fn make_encoder_report(
  batch_size : Int,
  tokens : Int,
  width : Int,
  hidden_size : Int,
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
  cpu_checksum : Float,
  webnn_checksum : Float,
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (batchSize, tokens, width, hiddenSize, heads, maskMode, operationsPerSample, warmup, iterations, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnSamplesMs, cpuChecksum, webnnChecksum, maxAbsError) => ({
  #|   batchSize,
  #|   tokens,
  #|   width,
  #|   hiddenSize,
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
  #|   cpuChecksum,
  #|   webnnChecksum,
  #|   maxAbsError,
  #| })

///|
pub async fn transformer_encoder(
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if batch_size <= 0 || batch_size > 64 {
    raise @tensor.TensorError::new(
      "encoder batch size must be between 1 and 64",
    )
  }
  if tokens <= 0 || tokens > 256 {
    raise @tensor.TensorError::new(
      "encoder token count must be between 1 and 256",
    )
  }
  if width <= 0 || width > 512 {
    raise @tensor.TensorError::new("encoder width must be between 1 and 512")
  }
  if heads <= 0 || width % heads != 0 {
    raise @tensor.TensorError::new(
      "encoder heads must be positive and divide width",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("encoder warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("encoder iterations must be positive")
  }
  let hidden_size = width * 2
  let config = @model.TransformerEncoderConfig::new(
    width, heads, hidden_size, 3, 1.0e-5,
  )
  let mask = encoder_attention_mask(mask_mode, batch_size, tokens)
  let params = encoder_parameters(config, 0)
  let input_shape = @shape.Shape::new([batch_size, tokens, width])
  let inputs = encoder_inputs(batch_size, tokens, width)
  let operations_per_sample = encoder_operations_per_sample(
    batch_size, tokens, width,
  )

  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu_model = @cpu.materialize_transformer_encoder(config, params, mask)
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
  let webnn_model = builder.materialize_transformer_encoder(
    config, params, mask,
  )
  let output = webnn_model.forward(input_binding.tensor())
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
  make_encoder_report(
    batch_size,
    tokens,
    width,
    hidden_size,
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
    checksum(cpu_output),
    checksum(webnn_output),
    max_abs_error(cpu_output, webnn_output),
  )
}
