# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- **The standard WebNN bindings moved out of this repository.** `webnn/raw`
  was a thin FFI over the WebNN JavaScript API with no dependency on anything
  here, so it now lives in
  [`mizchi/js_web/nn`](https://github.com/mizchi/js.mbt/tree/main/modules/js_web/nn)
  and this repository consumes it. Everything that is not standard stays:
  `webnn/compat`, the LiteRT/TFLite parser and lowering, shape inference,
  BERT loading, and the runtime, program cache and execution pools.

  For anyone importing `mizchi/webnn/webnn/raw` directly, the replacement is
  `mizchi/js_web/nn`. The types and functions keep their names, with one
  change: `MLGraphBuilder::new(context)` is now `MLGraphBuilder(context)`,
  MoonBit's canonical constructor form.

- Dependencies are now `mizchi/js_core` and `mizchi/js_web` rather than the
  `mizchi/js` facade, which as of 0.13.0 is a convenience wrapper that would
  pull in the whole surface. `@js.Promise` and `@js.from_async` are
  `@core.Promise` and `@core.from_async`.

## [0.1.0] - 2026-07-15

Initial public release.

### Added

- A typed MoonBit facade for WebNN graph construction and execution.
- Fixed-shape `float32` operations for linear, convolutional, and Transformer workloads.
- Reusable programs, named multi-input/output graphs, execution pools, and lifecycle-safe caches.
- LiteRT IR lowering and TFLite parsing, execution, quantization, and CPU fallback APIs.
- Backend-neutral model definitions with CPU and WebNN implementations.
- BERT SafeTensors loading and parameter validation.
- English and Japanese documentation with a minimal inference example.
- Chrome Canary headless E2E coverage and benchmark playgrounds in the repository.

### Known limitations

- WebNN execution requires a compatible WebNN-enabled browser environment.
- Shapes are fixed at graph construction time.
- The primary graph API currently uses `float32`; quantized values are supported through the TFLite runtime APIs.
