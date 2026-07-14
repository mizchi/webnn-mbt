///|
extern "js" fn make_bert_encoder_report(
  layers : Int,
  tensor_count : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  hidden_size : Int,
  heads : Int,
  mask_mode : String,
  operations_per_sample : Int,
  warmup : Int,
  iterations : Int,
  load_ms : Double,
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
  #| (layers, tensorCount, batchSize, tokens, width, hiddenSize, heads, maskMode, operationsPerSample, warmup, iterations, loadMs, contextMs, graphMs, compileMs, prepareMs, cpuSamplesMs, webnnSamplesMs, cpuChecksum, webnnChecksum, maxAbsError) => ({
  #|   architecture: "bert-post-norm",
  #|   layers,
  #|   tensorCount,
  #|   batchSize,
  #|   tokens,
  #|   width,
  #|   hiddenSize,
  #|   heads,
  #|   maskMode,
  #|   operationsPerSample,
  #|   warmup,
  #|   iterations,
  #|   loadMs,
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
pub async fn bert_encoder(
  layers : Int,
  batch_size : Int,
  tokens : Int,
  width : Int,
  heads : Int,
  mask_mode : String,
  warmup : Int,
  iterations : Int,
) -> @core.Any {
  if layers <= 0 || layers > 16 {
    raise @tensor.TensorError::new(
      "BERT encoder layers must be between 1 and 16",
    )
  }
  if batch_size <= 0 || batch_size > 64 {
    raise @tensor.TensorError::new(
      "BERT encoder batch size must be between 1 and 64",
    )
  }
  if tokens <= 0 || tokens > 256 {
    raise @tensor.TensorError::new(
      "BERT encoder token count must be between 1 and 256",
    )
  }
  if width <= 0 || width > 512 {
    raise @tensor.TensorError::new(
      "BERT encoder width must be between 1 and 512",
    )
  }
  if heads <= 0 || width % heads != 0 {
    raise @tensor.TensorError::new(
      "BERT encoder heads must be positive and divide width",
    )
  }
  if warmup < 0 {
    raise @tensor.TensorError::new("BERT encoder warmup must be non-negative")
  }
  if iterations <= 0 {
    raise @tensor.TensorError::new("BERT encoder iterations must be positive")
  }
  let hidden_size = width * 2
  let config = @model.BertEncoderConfig::new(
    layers, width, heads, hidden_size, 3, 1.0e-12,
  )
  let checkpoint_tensors = @bert.tiny_hugging_face_encoder_tensors(
    config, "bert.encoder",
  )
  let load_start = performance_now()
  let parameters = @bert.load_hugging_face_encoder(
    config, checkpoint_tensors, "bert.encoder",
  )
  let load_ms = performance_now() - load_start
  let mask = encoder_attention_mask(mask_mode, batch_size, tokens)
  let input_shape = @shape.Shape::new([batch_size, tokens, width])
  let inputs = encoder_inputs(batch_size, tokens, width)
  let operations_per_sample = encoder_stack_operations_per_sample(
    layers, batch_size, tokens, width,
  )

  let cpu_input = @cpu.CpuTensor::new(input_shape, inputs)
  let cpu_model = @cpu.materialize_bert_encoder_stack(config, parameters, mask)
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
  let webnn_model = builder.materialize_bert_encoder_stack(
    config, parameters, mask,
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
  make_bert_encoder_report(
    layers,
    checkpoint_tensors.length(),
    batch_size,
    tokens,
    width,
    hidden_size,
    heads,
    mask_mode,
    operations_per_sample,
    warmup,
    iterations,
    load_ms,
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
