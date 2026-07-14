///|
fn benchmark_mnist_cache(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @mnist.benchmark_cache(batch_size, warmup, iterations)
  })
}

///|
fn run_mnist_cached(batch_size : Int) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.run_cached(batch_size) })
}

///|
fn evaluate_mnist(count : Int) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.evaluate(count) })
}

///|
fn benchmark_mnist(
  batch_size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @mnist.benchmark(batch_size, warmup, iterations) })
}

///|
fn benchmark_matmul(
  size : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() { @bench.matmul(size, warmup, iterations) })
}

///|
fn benchmark_transformer_block(
  tokens : Int,
  width : Int,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_block(tokens, width, warmup, iterations)
  })
}

///|
fn benchmark_self_attention(
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.self_attention(tokens, width, heads, mask_mode, warmup, iterations)
  })
}

///|
fn benchmark_transformer_encoder(
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_encoder(
      batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
fn benchmark_transformer_encoder_stack(
  layers : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.transformer_encoder_stack(
      layers, batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
fn benchmark_bert_encoder(
  layers : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    @bench.bert_encoder(
      layers, batch_size, tokens, width, heads, mask_mode, warmup, iterations,
    )
  })
}

///|
extern "js" fn make_bert_safetensors_fixture_report(
  file_bytes : Int,
  tensor_count : Int,
  layers : Int,
  input_shape : Array[Int],
  output_shape : Array[Int],
  cpu_output : Array[Float],
  webnn_output : Array[Float],
  maximum_absolute_error : Float,
) -> @core.Any =
  #| (fileBytes, tensorCount, layers, inputShape, outputShape, cpuOutput, webnnOutput, maxAbsError) => ({
  #|   fileBytes,
  #|   tensorCount,
  #|   layers,
  #|   inputShape,
  #|   outputShape,
  #|   cpuOutput,
  #|   webnnOutput,
  #|   maxAbsError,
  #| })

///|
fn bert_fixture_max_abs_error(
  expected : Array[Float],
  actual : Array[Float],
) -> Float {
  let mut maximum : Float = 0.0
  for index, value in expected {
    let difference = (value - actual[index]).abs()
    if difference > maximum {
      maximum = difference
    }
  }
  maximum
}

///|
fn run_bert_safetensors_fixture() -> @js.Promise[@core.Any] {
  @js.from_async(async fn() {
    let bytes = fetch_tflite_fixture(
      "/fixtures/bert/tiny-bert-encoder.safetensors",
    ).wait()
    let tensors = @bert.parse_safetensors(bytes)
    let config = @model.BertEncoderConfig::new(2, 8, 2, 16, 3, 1.0e-12)
    let parameters = @bert.load_hugging_face_encoder(
      config, tensors, "bert.encoder",
    )
    let input_shape = @shape.Shape::new([2, 4, 8])
    let inputs = Array::makei(input_shape.element_count(), fn(index) {
      Float::from_int((index * 7 + index / 5) % 37 - 18) / Float::from_int(16)
    })
    let mask = @model.AttentionMask::batched_padding(
      [[true, true, true, true], [true, true, true, false]],
      -10000.0,
    )
    let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
    let cpu_output = @cpu.materialize_bert_encoder_stack(
        config,
        parameters,
        Some(mask),
      )
      .forward(cpu_input)
      .values()
    let builder = @webnn.WebNNGraphBuilder::new(@compat.DevicePreference::Npu)
    defer builder.destroy()
    let input_binding = builder.input("input", input_shape)
    let output = builder
      .materialize_bert_encoder_stack(config, parameters, Some(mask))
      .forward(input_binding.tensor())
    let session = builder.compile_single(input_binding, "output", output)
    let webnn_output = session.run(inputs)
    make_bert_safetensors_fixture_report(
      bytes.length(),
      tensors.length(),
      config.layers(),
      input_shape.dimensions(),
      output.shape().dimensions(),
      cpu_output,
      webnn_output,
      bert_fixture_max_abs_error(cpu_output, webnn_output),
    )
  })
}
