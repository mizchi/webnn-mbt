///|
pub(all) struct Mlp[T] {
  hidden : Linear[T]
  output : Linear[T]
}

///|
pub fn[T : @tensor.TensorOps] Mlp::forward(self : Mlp[T], input : T) -> T raise {
  let hidden = self.hidden.forward(input).relu()
  self.output.forward(hidden)
}

///|
pub(all) struct MlpParams {
  weight1 : Array[Float]
  bias1 : Array[Float]
  weight2 : Array[Float]
  bias2 : Array[Float]
}
