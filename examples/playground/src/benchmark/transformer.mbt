///|
priv struct TransformerParameters {
  input_weight : Array[Float]
  input_bias : Array[Float]
  normalization_scale : Array[Float]
  normalization_bias : Array[Float]
  output_weight : Array[Float]
  output_bias : Array[Float]
}

///|
fn transformer_parameters(
  width : Int,
  hidden_size : Int,
) -> TransformerParameters {
  transformer_parameters_with_offset(width, hidden_size, 0)
}

///|
fn transformer_parameters_with_offset(
  width : Int,
  hidden_size : Int,
  offset : Int,
) -> TransformerParameters {
  {
    input_weight: Array::makei(width * hidden_size, fn(index) {
      Float::from_int((index * 5 + 3 + offset * 7) % 23 - 11) /
      Float::from_int(32)
    }),
    input_bias: Array::makei(hidden_size, fn(index) {
      Float::from_int((index + offset * 2) % 7 - 3) / Float::from_int(64)
    }),
    normalization_scale: Array::makei(hidden_size, fn(index) {
      Float::from_int(12 + (index + offset) % 9) / Float::from_int(16)
    }),
    normalization_bias: Array::makei(hidden_size, fn(index) {
      Float::from_int((index + offset) % 5 - 2) / Float::from_int(64)
    }),
    output_weight: Array::makei(hidden_size * width, fn(index) {
      Float::from_int((index * 7 + 1 + offset * 11) % 29 - 14) /
      Float::from_int(64)
    }),
    output_bias: Array::makei(width, fn(index) {
      Float::from_int((index + offset * 3) % 7 - 3) / Float::from_int(128)
    }),
  }
}

///|
fn transformer_feed_forward_parameters_with_offset(
  config : @model.TransformerEncoderConfig,
  offset : Int,
) -> @model.FeedForwardParameters raise @tensor.TensorError {
  let params = transformer_parameters_with_offset(
    config.width(),
    config.hidden_size(),
    offset,
  )
  @model.FeedForwardParameters::new(
    config,
    params.input_weight,
    params.input_bias,
    params.output_weight,
    params.output_bias,
  )
}

///|
fn transformer_inputs(tokens : Int, width : Int) -> Array[Float] {
  Array::makei(tokens * width, fn(index) {
    Float::from_int((index * 3 + index / 5) % 31 - 15) / Float::from_int(16)
  })
}

///|
fn transformer_operations_per_sample(tokens : Int, width : Int) -> Int {
  let multiply_accumulates = tokens * width * width * 4
  if multiply_accumulates <= 65536 {
    5
  } else if multiply_accumulates <= 262144 {
    3
  } else {
    1
  }
}

///|
fn transformer_cpu_model(
  params : TransformerParameters,
  width : Int,
  hidden_size : Int,
  normalization_axis : Int,
) -> @model.TransformerBlock[@cpu.CpuTensor] raise {
  let hidden_shape = @shape.Shape::new([hidden_size])
  {
    input_projection: {
      weight: @cpu.CpuTensor::new(
        @shape.Shape::new([width, hidden_size]),
        params.input_weight,
      ),
      bias: @cpu.CpuTensor::new(hidden_shape, params.input_bias),
    },
    normalization_scale: @cpu.CpuTensor::new(
      hidden_shape,
      params.normalization_scale,
    ),
    normalization_bias: @cpu.CpuTensor::new(
      hidden_shape,
      params.normalization_bias,
    ),
    output_projection: {
      weight: @cpu.CpuTensor::new(
        @shape.Shape::new([hidden_size, width]),
        params.output_weight,
      ),
      bias: @cpu.CpuTensor::new(@shape.Shape::new([width]), params.output_bias),
    },
    normalization_axes: [normalization_axis],
    epsilon: 1.0e-5,
  }
}

///|
fn transformer_webnn_model(
  builder : @webnn.WebNNGraphBuilder,
  params : TransformerParameters,
  width : Int,
  hidden_size : Int,
  normalization_axis : Int,
) -> @model.TransformerBlock[@webnn.WebNNTensor] raise {
  let hidden_shape = @shape.Shape::new([hidden_size])
  {
    input_projection: {
      weight: builder.constant(
        @shape.Shape::new([width, hidden_size]),
        params.input_weight,
      ),
      bias: builder.constant(hidden_shape, params.input_bias),
    },
    normalization_scale: builder.constant(
      hidden_shape,
      params.normalization_scale,
    ),
    normalization_bias: builder.constant(
      hidden_shape,
      params.normalization_bias,
    ),
    output_projection: {
      weight: builder.constant(
        @shape.Shape::new([hidden_size, width]),
        params.output_weight,
      ),
      bias: builder.constant(@shape.Shape::new([width]), params.output_bias),
    },
    normalization_axes: [normalization_axis],
    epsilon: 1.0e-5,
  }
}

///|
extern "js" fn make_transformer_report(
  tokens : Int,
  width : Int,
  hidden_size : Int,
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
  #| (tokens, width, hiddenSize, operationsPerSample, warmup, iterations, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnSamplesMs, cpuChecksum, webnnChecksum, maxAbsError) => ({
  #|   tokens,
  #|   width,
  #|   hiddenSize,
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
pub async fn transformer_block(
  tokens : Int,
  width : Int,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if tokens <= 0 || tokens > 256 {
    raise @tensor.TensorError::new(
      "Transformer token count must be between 1 and 256",
    )
  }
  if width <= 0 || width > 512 {
    raise @tensor.TensorError::new(
      "Transformer width must be between 1 and 512",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("Transformer warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("Transformer iterations must be positive")
  }
  let hidden_size = width * 2
  let params = transformer_parameters(width, hidden_size)
  let input_shape = @shape.Shape::new([tokens, width])
  let inputs = transformer_inputs(tokens, width)
  let operations_per_sample = transformer_operations_per_sample(tokens, width)

  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu_model = transformer_cpu_model(params, width, hidden_size, 1)
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
  let output = transformer_webnn_model(builder, params, width, hidden_size, 1).forward(
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
  make_transformer_report(
    tokens,
    width,
    hidden_size,
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
