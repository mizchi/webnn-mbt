///|
let input_size : Int = 784

///|
let hidden_size : Int = 128

///|
let output_size : Int = 10

///|
pub struct Fixture {
  params : @model.MlpParams
  count : Int
  images : Array[Float]
  labels : Array[Int]
}

///|
extern "js" fn fetch_bytes(url : String) -> @js.Promise[Bytes] =
  #| async (url) => {
  #|   const response = await fetch(url);
  #|   if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  #|   return new Uint8Array(await response.arrayBuffer());
  #| }

///|
fn read_u32_le(bytes : Bytes, offset : Int) -> UInt {
  for index = 0, value = UInt::default()
      index < 4
      index = index + 1, value = value |
        (bytes[offset + index].to_uint() << (index * 8)) {

  } nobreak {
    value
  }
}

///|
fn read_u32_be(bytes : Bytes, offset : Int) -> UInt {
  for index = 0, value = UInt::default()
      index < 4
      index = index + 1, value = value |
        (bytes[offset + index].to_uint() << ((3 - index) * 8)) {

  } nobreak {
    value
  }
}

///|
fn floats_from(bytes : Bytes, offset : Int, count : Int) -> Array[Float] {
  Array::makei(count, fn(index) {
    Float::reinterpret_from_uint(read_u32_le(bytes, offset + index * 4))
  })
}

///|
fn parse_params(bytes : Bytes) -> @model.MlpParams raise @tensor.TensorError {
  let weight1_count = input_size * hidden_size
  let bias1_count = hidden_size
  let weight2_count = hidden_size * output_size
  let bias2_count = output_size
  let expected = (weight1_count + bias1_count + weight2_count + bias2_count) * 4
  if bytes.length() != expected {
    raise @tensor.TensorError::new(
      "MNIST weight bytes: expected {expected}, received {bytes.length()}",
    )
  }
  let mut offset = 0
  let weight1 = floats_from(bytes, offset, weight1_count)
  offset = offset + weight1_count * 4
  let bias1 = floats_from(bytes, offset, bias1_count)
  offset = offset + bias1_count * 4
  let weight2 = floats_from(bytes, offset, weight2_count)
  offset = offset + weight2_count * 4
  let bias2 = floats_from(bytes, offset, bias2_count)
  { weight1, bias1, weight2, bias2 }
}

///|
fn parse_images(bytes : Bytes) -> (Int, Array[Float]) raise @tensor.TensorError {
  if bytes.length() < 16 || read_u32_be(bytes, 0).reinterpret_as_int() != 2051 {
    raise @tensor.TensorError::new("invalid MNIST image IDX header")
  }
  let count = read_u32_be(bytes, 4).reinterpret_as_int()
  let rows = read_u32_be(bytes, 8).reinterpret_as_int()
  let columns = read_u32_be(bytes, 12).reinterpret_as_int()
  let expected = 16 + count * rows * columns
  if rows * columns != input_size || bytes.length() != expected {
    raise @tensor.TensorError::new("invalid MNIST image IDX shape or length")
  }
  let denominator = Float::from_int(255)
  let images = Array::makei(count * input_size, fn(index) {
    Float::from_int(bytes[16 + index].to_int()) / denominator
  })
  (count, images)
}

///|
fn parse_labels(bytes : Bytes) -> (Int, Array[Int]) raise @tensor.TensorError {
  if bytes.length() < 8 || read_u32_be(bytes, 0).reinterpret_as_int() != 2049 {
    raise @tensor.TensorError::new("invalid MNIST label IDX header")
  }
  let count = read_u32_be(bytes, 4).reinterpret_as_int()
  if bytes.length() != 8 + count {
    raise @tensor.TensorError::new("invalid MNIST label IDX length")
  }
  let labels = Array::makei(count, fn(index) { bytes[8 + index].to_int() })
  (count, labels)
}

///|
pub async fn load_fixture() -> Fixture {
  let resources = @js.Promise::all([
    fetch_bytes("/fixtures/mnist/mlp_784_128_10.bin"),
    fetch_bytes("/fixtures/mnist/t10k-images-idx3-ubyte"),
    fetch_bytes("/fixtures/mnist/t10k-labels-idx1-ubyte"),
  ]).wait()
  let params = parse_params(resources[0])
  let (image_count, images) = parse_images(resources[1])
  let (label_count, labels) = parse_labels(resources[2])
  if image_count != label_count {
    raise @tensor.TensorError::new(
      "MNIST fixture count mismatch: images={image_count}, labels={label_count}",
    )
  }
  { params, count: image_count, images, labels }
}

///|
pub fn Fixture::inputs(
  self : Fixture,
  count : Int,
) -> Array[Float] raise @tensor.TensorError {
  if count <= 0 || count > self.count {
    raise @tensor.TensorError::new(
      "MNIST count must be between 1 and {self.count}",
    )
  }
  Array::makei(count * input_size, fn(index) { self.images[index] })
}

///|
pub fn Fixture::expected_labels(
  self : Fixture,
  count : Int,
) -> Array[Int] raise @tensor.TensorError {
  if count <= 0 || count > self.count {
    raise @tensor.TensorError::new(
      "MNIST count must be between 1 and {self.count}",
    )
  }
  Array::makei(count, fn(index) { self.labels[index] })
}
