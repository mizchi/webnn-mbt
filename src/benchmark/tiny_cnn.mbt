///|
let tiny_input_size : Int = 8

///|
let tiny_conv_channels : Int = 4

///|
let tiny_feature_size : Int = 4 * 6 * 6

///|
let tiny_classes : Int = 10

///|
priv struct TinyCnnParameters {
  conv_weight : Array[Float]
  conv_bias : Array[Float]
  classifier_weight : Array[Float]
  classifier_bias : Array[Float]
}

///|
fn tiny_cnn_parameters() -> TinyCnnParameters {
  {
    conv_weight: Array::makei(tiny_conv_channels * 3 * 3, fn(index) {
      Float::from_int((index * 5 + 3) % 11 - 5) / Float::from_int(16)
    }),
    conv_bias: Array::makei(tiny_conv_channels, fn(index) {
      Float::from_int(index - 2) / Float::from_int(32)
    }),
    classifier_weight: Array::makei(tiny_feature_size * tiny_classes, fn(
      index,
    ) {
      Float::from_int((index * 7 + 1) % 17 - 8) / Float::from_int(64)
    }),
    classifier_bias: Array::makei(tiny_classes, fn(index) {
      Float::from_int(index - 5) / Float::from_int(64)
    }),
  }
}

///|
fn tiny_cnn_inputs(batch_size : Int) -> Array[Float] {
  Array::makei(batch_size * tiny_input_size * tiny_input_size, fn(index) {
    Float::from_int((index * 3 + index / 7) % 29 - 14) / Float::from_int(16)
  })
}

///|
fn tiny_input_shape(
  batch_size : Int,
  input_layout : @shape.InputLayout,
) -> @shape.Shape raise {
  match input_layout {
    @shape.InputLayout::Nchw =>
      @shape.Shape::new([batch_size, 1, tiny_input_size, tiny_input_size])
    @shape.InputLayout::Nhwc =>
      @shape.Shape::new([batch_size, tiny_input_size, tiny_input_size, 1])
  }
}

///|
fn tiny_conv_filter_shape(
  input_layout : @shape.InputLayout,
) -> @shape.Shape raise {
  match input_layout {
    @shape.InputLayout::Nchw => @shape.Shape::new([tiny_conv_channels, 1, 3, 3])
    @shape.InputLayout::Nhwc => @shape.Shape::new([3, 3, 1, tiny_conv_channels])
  }
}

///|
fn tiny_conv_bias_shape(
  input_layout : @shape.InputLayout,
) -> @shape.Shape raise {
  match input_layout {
    @shape.InputLayout::Nchw => @shape.Shape::new([1, tiny_conv_channels, 1, 1])
    @shape.InputLayout::Nhwc => @shape.Shape::new([1, 1, 1, tiny_conv_channels])
  }
}

///|
fn tiny_filter_layout(
  input_layout : @shape.InputLayout,
) -> @shape.Conv2dFilterLayout {
  match input_layout {
    @shape.InputLayout::Nchw => @shape.Conv2dFilterLayout::Oihw
    @shape.InputLayout::Nhwc => @shape.Conv2dFilterLayout::Hwio
  }
}

///|
fn tiny_conv_options(
  input_layout : @shape.InputLayout,
) -> @shape.Conv2dOptions raise {
  @shape.Conv2dOptions::new_with_layout(
    [0, 0, 0, 0],
    [1, 1],
    [1, 1],
    1,
    input_layout,
    tiny_filter_layout(input_layout),
  )
}

///|
fn tiny_conv_weight(
  params : TinyCnnParameters,
  input_layout : @shape.InputLayout,
) -> Array[Float] {
  match input_layout {
    @shape.InputLayout::Nchw => params.conv_weight.copy()
    @shape.InputLayout::Nhwc => {
      let values = Array::make(params.conv_weight.length(), Float::from_int(0))
      for output_channel in 0..<tiny_conv_channels {
        for filter_y in 0..<3 {
          for filter_x in 0..<3 {
            let source_index = (output_channel * 3 + filter_y) * 3 + filter_x
            let destination_index = (filter_y * 3 + filter_x) *
              tiny_conv_channels +
              output_channel
            values[destination_index] = params.conv_weight[source_index]
          }
        }
      }
      values
    }
  }
}

///|
fn tiny_classifier_weight(
  params : TinyCnnParameters,
  input_layout : @shape.InputLayout,
) -> Array[Float] {
  match input_layout {
    @shape.InputLayout::Nchw => params.classifier_weight.copy()
    @shape.InputLayout::Nhwc => {
      let values = Array::make(
        params.classifier_weight.length(),
        Float::from_int(0),
      )
      for channel in 0..<tiny_conv_channels {
        for y in 0..<6 {
          for x in 0..<6 {
            let source_feature = (channel * 6 + y) * 6 + x
            let destination_feature = (y * 6 + x) * tiny_conv_channels + channel
            for class_index in 0..<tiny_classes {
              values[destination_feature * tiny_classes + class_index] = params.classifier_weight[source_feature *
                tiny_classes +
                class_index]
            }
          }
        }
      }
      values
    }
  }
}

///|
fn tiny_cnn_operations_per_sample(batch_size : Int) -> Int {
  if batch_size == 1 {
    5
  } else if batch_size <= 4 {
    3
  } else {
    1
  }
}

///|
fn tiny_cnn_cpu_model(
  params : TinyCnnParameters,
  input_layout : @shape.InputLayout,
) -> @model.TinyCnn[@cpu.CpuTensor] raise {
  {
    convolution: {
      weight: @cpu.CpuTensor::new(
        tiny_conv_filter_shape(input_layout),
        tiny_conv_weight(params, input_layout),
      ),
      bias: @cpu.CpuTensor::new(
        tiny_conv_bias_shape(input_layout),
        params.conv_bias,
      ),
      options: tiny_conv_options(input_layout),
    },
    classifier: {
      weight: @cpu.CpuTensor::new(
        @shape.Shape::new([tiny_feature_size, tiny_classes]),
        tiny_classifier_weight(params, input_layout),
      ),
      bias: @cpu.CpuTensor::new(
        @shape.Shape::new([tiny_classes]),
        params.classifier_bias,
      ),
    },
  }
}

///|
fn tiny_cnn_webnn_model(
  builder : @webnn.WebNNGraphBuilder,
  params : TinyCnnParameters,
  input_layout : @shape.InputLayout,
) -> @model.TinyCnn[@webnn.WebNNTensor] raise {
  {
    convolution: {
      weight: builder.constant(
        tiny_conv_filter_shape(input_layout),
        tiny_conv_weight(params, input_layout),
      ),
      bias: builder.constant(
        tiny_conv_bias_shape(input_layout),
        params.conv_bias,
      ),
      options: tiny_conv_options(input_layout),
    },
    classifier: {
      weight: builder.constant(
        @shape.Shape::new([tiny_feature_size, tiny_classes]),
        tiny_classifier_weight(params, input_layout),
      ),
      bias: builder.constant(
        @shape.Shape::new([tiny_classes]),
        params.classifier_bias,
      ),
    },
  }
}

///|
fn tiny_predictions(
  probabilities : Array[Float],
  batch_size : Int,
) -> Array[Int] {
  Array::makei(batch_size, fn(batch) {
    let offset = batch * tiny_classes
    let mut best_index = 0
    let mut best_value = probabilities[offset]
    for index in 1..<tiny_classes {
      let value = probabilities[offset + index]
      if value > best_value {
        best_value = value
        best_index = index
      }
    }
    best_index
  })
}

///|
fn tiny_prediction_matches(lhs : Array[Int], rhs : Array[Int]) -> Int {
  let mut matches = 0
  for index, value in lhs {
    if value == rhs[index] {
      matches = matches + 1
    }
  }
  matches
}

///|
fn maximum_probability_sum_error(
  probabilities : Array[Float],
  batch_size : Int,
) -> Float {
  let mut maximum = Float::from_int(0)
  for batch in 0..<batch_size {
    let mut sum = Float::from_int(0)
    for class_index in 0..<tiny_classes {
      sum = sum + probabilities[batch * tiny_classes + class_index]
    }
    let difference = sum - Float::from_int(1)
    let absolute = if difference < 0.0 { -difference } else { difference }
    if absolute > maximum {
      maximum = absolute
    }
  }
  maximum
}

///|
extern "js" fn make_tiny_cnn_report(
  input_layout : String,
  batch_size : Int,
  operations_per_sample : Int,
  warmup : Int,
  iterations : Int,
  context_ms : Double,
  graph_ms : Double,
  compile_ms : Double,
  prepare_ms : Double,
  cpu_samples_ms : Array[Double],
  webnn_samples_ms : Array[Double],
  prediction_matches : Int,
  maximum_absolute_error : Float,
  maximum_probability_sum_error : Float,
) -> @core.Any =
  #| (inputLayout, batchSize, operationsPerSample, warmup, iterations, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnSamplesMs, predictionMatches, maxAbsError, maxProbabilitySumError) => ({
  #|   inputLayout,
  #|   batchSize,
  #|   operationsPerSample,
  #|   warmup,
  #|   iterations,
  #|   contextMs,
  #|   graphMs,
  #|   compileMs,
  #|   prepareMs,
  #|   cpuSamplesMs,
  #|   webnnSamplesMs,
  #|   predictionMatches,
  #|   maxAbsError,
  #|   maxProbabilitySumError,
  #| })

///|
pub async fn tiny_cnn(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  tiny_cnn_with_layout(batch_size, warmup, iterations, @shape.InputLayout::Nchw)
}

///|
pub async fn tiny_cnn_with_layout(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
  input_layout : @shape.InputLayout,
) -> @core.Any {
  if batch_size <= 0 || batch_size > 256 {
    raise @tensor.TensorError::new(
      "Tiny CNN batch size must be between 1 and 256",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("Tiny CNN warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("Tiny CNN iterations must be positive")
  }
  let params = tiny_cnn_parameters()
  let input_shape = tiny_input_shape(batch_size, input_layout)
  let inputs = tiny_cnn_inputs(batch_size)
  let operations_per_sample = tiny_cnn_operations_per_sample(batch_size)
  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu_model = tiny_cnn_cpu_model(params, input_layout)
  let mut cpu_probabilities : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      cpu_probabilities = cpu_model.forward(cpu_input).values()
    }
  }
  let cpu_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      cpu_probabilities = cpu_model.forward(cpu_input).values()
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
  let output = tiny_cnn_webnn_model(builder, params, input_layout).forward(
    input_binding.tensor(),
  )
  let graph_ms = performance_now() - graph_start
  let compile_start = performance_now()
  let session = builder.compile_single(input_binding, "probabilities", output)
  let compile_ms = performance_now() - compile_start
  let prepare_start = performance_now()
  let execution = session.prepare()
  let prepare_ms = performance_now() - prepare_start
  defer execution.destroy()
  let mut webnn_probabilities : Array[Float] = []
  for _ in 0..<warmup {
    for _ in 0..<operations_per_sample {
      webnn_probabilities = execution.run(inputs)
    }
  }
  let webnn_samples_ms : Array[Double] = []
  for _ in 0..<iterations {
    let start = performance_now()
    for _ in 0..<operations_per_sample {
      webnn_probabilities = execution.run(inputs)
    }
    webnn_samples_ms.push(
      (performance_now() - start) / Double::from_int(operations_per_sample),
    )
  }
  make_tiny_cnn_report(
    input_layout.to_webnn_string(),
    batch_size,
    operations_per_sample,
    warmup,
    iterations,
    context_ms,
    graph_ms,
    compile_ms,
    prepare_ms,
    cpu_samples_ms,
    webnn_samples_ms,
    tiny_prediction_matches(
      tiny_predictions(cpu_probabilities, batch_size),
      tiny_predictions(webnn_probabilities, batch_size),
    ),
    max_abs_error(cpu_probabilities, webnn_probabilities),
    maximum_probability_sum_error(webnn_probabilities, batch_size).max(
      maximum_probability_sum_error(cpu_probabilities, batch_size),
    ),
  )
}
