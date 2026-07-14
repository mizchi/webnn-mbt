///|
pub(all) enum DevicePreference {
  Cpu
  Gpu
  Npu
} derive(Debug, Eq)

///|
pub fn DevicePreference::to_webnn_string(self : DevicePreference) -> String {
  match self {
    Cpu => "cpu"
    Gpu => "gpu"
    Npu => "npu"
  }
}

///|
pub fn compatibility_mode() -> String {
  if @raw.uses_accelerated_contract() {
    "accelerated"
  } else {
    "legacy-device-type"
  }
}

///|
pub async fn create_context(preference : DevicePreference) -> @raw.MLContext {
  guard @raw.ml() is Some(ml) else {
    raise @tensor.TensorError::new(
      "WebNN is unavailable: navigator.ml is missing",
    )
  }
  let options = @raw.context_options(
    @raw.uses_accelerated_contract(),
    preference.to_webnn_string(),
  )
  ml.create_context(options)
}
