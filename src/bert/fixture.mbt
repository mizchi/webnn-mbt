///|
fn fixture_tensor(
  name : String,
  dimensions : Array[Int],
  values : Array[Float],
) -> NamedTensor raise @tensor.TensorError {
  let shape = @shape.Shape::new(dimensions) catch {
    error => raise @tensor.TensorError::new(error.to_string())
  }
  NamedTensor::new(name, shape, values)
}

///|
fn fixture_matrix(
  name : String,
  rows : Int,
  columns : Int,
  seed : Int,
) -> NamedTensor raise @tensor.TensorError {
  fixture_tensor(
    name,
    [rows, columns],
    Array::makei(rows * columns, fn(index) {
      Float::from_int((index * 13 + seed * 7) % 23 - 11) / Float::from_int(64)
    }),
  )
}

///|
fn fixture_bias(
  name : String,
  size : Int,
  seed : Int,
) -> NamedTensor raise @tensor.TensorError {
  fixture_tensor(
    name,
    [size],
    Array::makei(size, fn(index) {
      Float::from_int((index * 3 + seed) % 9 - 4) / Float::from_int(256)
    }),
  )
}

///|
fn fixture_normalization_scale(
  name : String,
  size : Int,
  seed : Int,
) -> NamedTensor raise @tensor.TensorError {
  fixture_tensor(
    name,
    [size],
    Array::makei(size, fn(index) {
      Float::from_int(1) +
      Float::from_int((index + seed) % 5 - 2) / Float::from_int(64)
    }),
  )
}

///|
/// Deterministic Hugging Face-style BERT encoder tensors for differential
/// tests and benchmarks. Matrices intentionally use checkpoint `[out, in]`
/// layout so every run exercises the production loader's transpose path.
pub fn tiny_hugging_face_encoder_tensors(
  config : @model.BertEncoderConfig,
  encoder_prefix : String,
) -> Array[NamedTensor] raise @tensor.TensorError {
  let width = config.width()
  let intermediate_size = config.intermediate_size()
  let tensors : Array[NamedTensor] = []
  for layer in 0..<config.layers() {
    let prefix = encoder_prefix + ".layer." + layer.to_string()
    let seed = layer * 17
    tensors.push(
      fixture_matrix(
        prefix + ".attention.self.query.weight",
        width,
        width,
        seed,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".attention.self.query.bias", width, seed),
    )
    tensors.push(
      fixture_matrix(
        prefix + ".attention.self.key.weight",
        width,
        width,
        seed + 1,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".attention.self.key.bias", width, seed + 1),
    )
    tensors.push(
      fixture_matrix(
        prefix + ".attention.self.value.weight",
        width,
        width,
        seed + 2,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".attention.self.value.bias", width, seed + 2),
    )
    tensors.push(
      fixture_matrix(
        prefix + ".attention.output.dense.weight",
        width,
        width,
        seed + 3,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".attention.output.dense.bias", width, seed + 3),
    )
    tensors.push(
      fixture_normalization_scale(
        prefix + ".attention.output.LayerNorm.weight",
        width,
        seed,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".attention.output.LayerNorm.bias", width, seed + 4),
    )
    tensors.push(
      fixture_matrix(
        prefix + ".intermediate.dense.weight",
        intermediate_size,
        width,
        seed + 5,
      ),
    )
    tensors.push(
      fixture_bias(
        prefix + ".intermediate.dense.bias",
        intermediate_size,
        seed + 5,
      ),
    )
    tensors.push(
      fixture_matrix(
        prefix + ".output.dense.weight",
        width,
        intermediate_size,
        seed + 6,
      ),
    )
    tensors.push(fixture_bias(prefix + ".output.dense.bias", width, seed + 6))
    tensors.push(
      fixture_normalization_scale(
        prefix + ".output.LayerNorm.weight",
        width,
        seed + 1,
      ),
    )
    tensors.push(
      fixture_bias(prefix + ".output.LayerNorm.bias", width, seed + 7),
    )
  }
  tensors
}
