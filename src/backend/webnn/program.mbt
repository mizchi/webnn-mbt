///|
pub struct WebNNProgram {
  owner : WebNNGraphBuilder
  executions : Array[WebNNExecution]
  scheduler : WebNNProgramScheduler
}

///| Compiles and prepares a reusable single-slot program. On success the

///|
/// returned program owns `self`; callers must not destroy the builder.
pub async fn WebNNGraphBuilder::compile_program_single(
  self : WebNNGraphBuilder,
  input : WebNNInput,
  output_name : String,
  output : WebNNTensor,
) -> WebNNProgram {
  self.compile_program_pool_single(input, output_name, output, 1)
}

///|
pub async fn WebNNGraphBuilder::compile_program_named(
  self : WebNNGraphBuilder,
  inputs : Array[WebNNInput],
  outputs : Array[WebNNOutput],
) -> WebNNProgram {
  self.compile_program_pool_named(inputs, outputs, 1)
}

///| Compiles one graph and prepares independent input/output tensor slots.

///|
/// On success the returned program owns `self`.
pub async fn WebNNGraphBuilder::compile_program_pool_single(
  self : WebNNGraphBuilder,
  input : WebNNInput,
  output_name : String,
  output : WebNNTensor,
  pool_size : Int,
) -> WebNNProgram {
  let named_output = self.output(output_name, output) catch {
    error => {
      self.destroy()
      raise error
    }
  }
  self.compile_program_pool_named([input], [named_output], pool_size)
}

///| Compiles one named graph and prepares independent I/O tensor slots.

///|
/// On success the returned program owns `self`.
pub async fn WebNNGraphBuilder::compile_program_pool_named(
  self : WebNNGraphBuilder,
  inputs : Array[WebNNInput],
  outputs : Array[WebNNOutput],
  pool_size : Int,
) -> WebNNProgram {
  if pool_size <= 0 {
    self.destroy()
    raise @tensor.TensorError::new(
      "WebNN program execution pool size must be positive",
    )
  }
  let session = self.compile_named(inputs, outputs) catch {
    error => {
      self.destroy()
      raise error
    }
  }
  let executions : Array[WebNNExecution] = []
  for _ in 0..<pool_size {
    let execution = session.prepare() catch {
      error => {
        executions.each(fn(prepared) { prepared.destroy() })
        self.destroy()
        raise error
      }
    }
    executions.push(execution)
  }
  { owner: self, executions, scheduler: new_program_scheduler(pool_size) }
}

///|
pub async fn WebNNProgram::run(
  self : WebNNProgram,
  input_values : Array[Float],
) -> Array[Float] {
  let execution = self.executions[0]
  if execution.input_specs.length() != 1 || execution.output_specs.length() != 1 {
    raise @tensor.TensorError::new(
      "single-value run requires exactly one input and one output",
    )
  }
  let results = self.run_named([
    { name_: execution.input_specs[0].name, values_: input_values },
  ])
  results[0].values_
}

///|
pub async fn WebNNProgram::run_named(
  self : WebNNProgram,
  input_values : Array[WebNNNamedValues],
) -> Array[WebNNNamedValues] {
  enqueue_program_run(self.scheduler, fn(slot) {
    @js.from_async(async fn() { self.executions[slot].run_named(input_values) })
  }).wait()
}

///|
pub fn WebNNProgram::pool_size(self : WebNNProgram) -> Int {
  self.executions.length()
}

///|
pub fn WebNNProgram::maximum_concurrency(self : WebNNProgram) -> Int {
  program_maximum_concurrency(self.scheduler)
}

///|
pub fn WebNNProgram::destroy(self : WebNNProgram) -> Unit {
  enqueue_program_destroy(self.scheduler, fn() {
    self.executions.each(fn(execution) { execution.destroy() })
    self.owner.destroy()
  })
}
