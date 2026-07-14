///|
pub(all) struct Conv2dLayer[T] {
  weight : T
  bias : T
  options : @shape.Conv2dOptions
}

///|
pub fn[T : @tensor.TensorOps] Conv2dLayer::forward(
  self : Conv2dLayer[T],
  input : T,
) -> T raise {
  input.conv2d(self.weight, self.options).add(self.bias)
}

///|
pub(all) struct TinyCnn[T] {
  convolution : Conv2dLayer[T]
  classifier : Linear[T]
}

///|
pub fn[T : @tensor.TensorOps] TinyCnn::forward(
  self : TinyCnn[T],
  input : T,
) -> T raise {
  let features = self.convolution.forward(input).relu()
  let feature_shape = features.shape()
  let batch_size = feature_shape.dimension(0)
  let flattened_size = feature_shape.element_count() / batch_size
  self.classifier
  .forward(features.reshape(@shape.Shape::new([batch_size, flattened_size])))
  .softmax(1)
}
