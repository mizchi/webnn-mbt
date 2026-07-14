///| High-level WebNN facade for the `mizchi/webnn` package.

///|
/// WebNN graph, execution, cache, and TFLite runtime APIs.
pub using @backend {
  type TfliteCpuRunner,
  type TfliteCpuRunnerCache,
  type TfliteCpuRunnerCacheMetrics,
  type TfliteModelArtifact,
  type TfliteRunner,
  type TfliteRunnerCache,
  type TfliteRunnerCacheKey,
  type TfliteRunnerCacheMetrics,
  type TfliteRunnerInput,
  type TfliteRunnerOutput,
  type TfliteRunnerValues,
  type WebNNCapabilities,
  type WebNNContextContract,
  type WebNNExecution,
  type WebNNGraphBuilder,
  type WebNNInput,
  type WebNNLoweredLiteRtGraph,
  type WebNNNamedValues,
  type WebNNOutput,
  type WebNNProgram,
  type WebNNProgramCache,
  type WebNNProgramCacheKey,
  type WebNNRuntime,
  type WebNNRuntimeFallback,
  type WebNNRuntimeMetrics,
  type WebNNSession,
  type WebNNTensor,
  probe_capabilities,
}

///|
/// Browser context selection and compatibility APIs.
pub using @compat {type DevicePreference, compatibility_mode}

///|
/// Validated tensor shapes and WebNN layout options.
pub using @shape {
  type Conv2dFilterLayout,
  type Conv2dOptions,
  type InputLayout,
  type Pool2dOptions,
  type Shape,
  type ShapeError,
}

///|
/// LiteRT IR, TFLite parsing, and quantization APIs.
pub using @litert {
  type LiteRtError,
  type LiteRtLoweredValue,
  type LiteRtModel,
  type LiteRtNode,
  type LiteRtValue,
  type TfliteModel,
  type TfliteRuntimeDataType,
  type TfliteRuntimeModel,
  type TfliteRuntimeTensor,
  dequantize_int8,
  dequantize_uint8,
  requantize_int8,
  requantize_uint8,
}

///|
/// Backend-neutral tensor contract and error type.
pub using @tensor {type TensorError, trait TensorOps}
