///|
pub(all) struct WebNNLoweredLiteRtGraph {
  inputs_ : Array[WebNNInput]
  outputs_ : Array[@litert.LiteRtLoweredValue[WebNNTensor]]
}

///|
pub fn WebNNLoweredLiteRtGraph::inputs(
  self : WebNNLoweredLiteRtGraph,
) -> Array[WebNNInput] {
  self.inputs_.copy()
}

///|
pub fn WebNNLoweredLiteRtGraph::outputs(
  self : WebNNLoweredLiteRtGraph,
) -> Array[@litert.LiteRtLoweredValue[WebNNTensor]] {
  self.outputs_.copy()
}

///|
pub fn WebNNGraphBuilder::lower_litert(
  self : WebNNGraphBuilder,
  model : @litert.LiteRtModel,
) -> WebNNLoweredLiteRtGraph raise {
  let inputs : Array[WebNNInput] = []
  let outputs = model.lower(
    (name, shape) => {
      let input = self.input(name, shape)
      inputs.push(input)
      input.tensor()
    },
    (_name, shape, values) => self.constant(shape, values),
  )
  { inputs_: inputs, outputs_: outputs }
}

///| Lower, compile, and prepare a reusable WebNN program for a LiteRT model.

///|
/// The returned program owns `self`; callers must not destroy the builder.
pub async fn WebNNGraphBuilder::compile_litert_program(
  self : WebNNGraphBuilder,
  model : @litert.LiteRtModel,
) -> WebNNProgram {
  self.compile_litert_program_pool(model, 1)
}

///| Lower, compile, and prepare a pooled WebNN program for a LiteRT model.

///|
/// The returned program owns `self`; a lowering or output-binding failure destroys it.
pub async fn WebNNGraphBuilder::compile_litert_program_pool(
  self : WebNNGraphBuilder,
  model : @litert.LiteRtModel,
  pool_size : Int,
) -> WebNNProgram {
  let lowered = self.lower_litert(model) catch {
    error => {
      self.destroy()
      raise error
    }
  }
  let outputs : Array[WebNNOutput] = []
  for lowered_output in lowered.outputs() {
    let output = self.output(lowered_output.name(), lowered_output.tensor()) catch {
      error => {
        self.destroy()
        raise error
      }
    }
    outputs.push(output)
  }
  self.compile_program_pool_named(lowered.inputs(), outputs, pool_size)
}
