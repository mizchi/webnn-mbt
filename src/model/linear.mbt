///|
pub(all) struct Linear[T] {
  weight : T
  bias : T
}

///|
pub fn[T : @tensor.TensorOps] Linear::forward(
  self : Linear[T],
  input : T,
) -> T raise {
  input.matmul(self.weight).add(self.bias)
}
