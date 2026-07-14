# Changelog

All notable changes to this project are documented in this file.

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
