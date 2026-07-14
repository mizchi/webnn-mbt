///|
fn cpu_model(params : @model.MlpParams) -> @model.Mlp[@cpu.CpuTensor] raise {
  {
    hidden: {
      weight: @cpu.CpuTensor::new(
        @shape.Shape::new([input_size, hidden_size]),
        params.weight1,
      ),
      bias: @cpu.CpuTensor::new(@shape.Shape::new([hidden_size]), params.bias1),
    },
    output: {
      weight: @cpu.CpuTensor::new(
        @shape.Shape::new([hidden_size, output_size]),
        params.weight2,
      ),
      bias: @cpu.CpuTensor::new(@shape.Shape::new([output_size]), params.bias2),
    },
  }
}

///|
fn webnn_model(
  builder : @webnn.WebNNGraphBuilder,
  params : @model.MlpParams,
) -> @model.Mlp[@webnn.WebNNTensor] raise {
  {
    hidden: {
      weight: builder.constant(
        @shape.Shape::new([input_size, hidden_size]),
        params.weight1,
      ),
      bias: builder.constant(@shape.Shape::new([hidden_size]), params.bias1),
    },
    output: {
      weight: builder.constant(
        @shape.Shape::new([hidden_size, output_size]),
        params.weight2,
      ),
      bias: builder.constant(@shape.Shape::new([output_size]), params.bias2),
    },
  }
}

///|
fn predictions(logits : Array[Float], count : Int) -> Array[Int] {
  Array::makei(count, fn(batch) {
    let offset = batch * output_size
    let mut best_index = 0
    let mut best_value = logits[offset]
    for index in 1..<output_size {
      let value = logits[offset + index]
      if value > best_value {
        best_value = value
        best_index = index
      }
    }
    best_index
  })
}

///|
fn max_abs_error(lhs : Array[Float], rhs : Array[Float]) -> Float {
  if lhs.length() != rhs.length() {
    return (1.0e30 : Float)
  }
  let mut maximum = Float::from_int(0)
  for index in 0..<lhs.length() {
    let difference = lhs[index] - rhs[index]
    let absolute = if difference < Float::from_int(0) {
      -difference
    } else {
      difference
    }
    if absolute > maximum {
      maximum = absolute
    }
  }
  maximum
}

///|
extern "js" fn make_evaluation_report(
  count : Int,
  cpu_correct : Int,
  webnn_correct : Int,
  prediction_matches : Int,
  max_abs_error : Float,
  labels : Array[Int],
  cpu_predictions : Array[Int],
  webnn_predictions : Array[Int],
  cpu_logits : Array[Float],
  webnn_logits : Array[Float],
) -> @core.Any =
  #| (count, cpuCorrect, webnnCorrect, predictionMatches, maxAbsError, labels, cpuPredictions, webnnPredictions, cpuLogits, webnnLogits) => ({
  #|   count,
  #|   cpuCorrect,
  #|   webnnCorrect,
  #|   predictionMatches,
  #|   maxAbsError,
  #|   labels,
  #|   cpuPredictions,
  #|   webnnPredictions,
  #|   cpuLogits,
  #|   webnnLogits,
  #| })

///|
pub async fn evaluate(count : Int) -> @core.Any {
  let fixture = load_fixture()
  let inputs = fixture.inputs(count)
  let labels = fixture.expected_labels(count)
  let input_shape = @shape.Shape::new([count, input_size])
  let cpu_logits = cpu_model(fixture.params)
    .forward(@cpu.CpuTensor::new(input_shape, inputs))
    .values()
  let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
  defer builder.destroy()
  let input_binding = builder.input("input", input_shape)
  let output = webnn_model(builder, fixture.params).forward(
    input_binding.tensor(),
  )
  let session = builder.compile_single(input_binding, "output", output)
  let webnn_logits = session.run(inputs)
  let cpu_predictions = predictions(cpu_logits, count)
  let webnn_predictions = predictions(webnn_logits, count)
  let mut cpu_correct = 0
  let mut webnn_correct = 0
  let mut prediction_matches = 0
  for index in 0..<count {
    if cpu_predictions[index] == labels[index] {
      cpu_correct = cpu_correct + 1
    }
    if webnn_predictions[index] == labels[index] {
      webnn_correct = webnn_correct + 1
    }
    if cpu_predictions[index] == webnn_predictions[index] {
      prediction_matches = prediction_matches + 1
    }
  }
  make_evaluation_report(
    count,
    cpu_correct,
    webnn_correct,
    prediction_matches,
    max_abs_error(cpu_logits, webnn_logits),
    labels,
    cpu_predictions,
    webnn_predictions,
    cpu_logits,
    webnn_logits,
  )
}
