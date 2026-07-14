# TFLite fixtures

`micro_speech_quantized.tflite` は TensorFlow Lite Micro の Micro Speech
（4分類 wake-word recognition）モデルです。手製FlatBufferではない実モデルで、
`CONV_2D`、`FULLY_CONNECTED`/`MATMUL`、`SOFTMAX` を含む経路を検証します。

- source: `tensorflow/tflite-micro` commit `fddd3707a3c5733af4cb866f18650441e6712504`
- license: Apache-2.0
- bytes: 18,800
- SHA-256: `09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83`

binaryは直接編集せず、完全性検証付きで再取得します。

```bash
just fixture-tflite
```

fixture の E2E はモデルを parse/lower/compile/run し、raw INT8 入出力を検証します。

`audio_preprocessor_int8.tflite` は同じ upstream の音声前処理モデルです。
最初の custom operator `SignalWindow` を名前付きで拒否する negative fixture として使い、
Signal Library / custom operator は現時点の WebNN lowerer の対象外であることを固定します。

- bytes: 8,772
- SHA-256: `278949d197166fb8b580c0bdc94e902fb709fec0569dcf5766816b28285440e5`

`person_detect.tflite` は同一commitの 96×96 grayscale 人物検出モデルです。Micro
Speech より深い量子化CNNを corpus に加え、実運用モデルが要求する operator を
検出・優先実装するために使います。

- bytes: 300,568
- SHA-256: `808cfdfc0cf3a6fa6f6fa26bfa379ea97c16d5db7334637766e39c3408502e9d`

`mobilenet_v2_1.0_224_inat_bird_quant.tflite` は Coral Edge TPU のテスト用
MobileNet V2量子化分類モデルです。parser対応範囲の次の優先演算を実モデルから選ぶ
probeとして使います（推論は Edge TPU delegateを前提にしません）。

- source: `google-coral/edgetpu` commit `5020de9386ff370dcc1f63291a2d0f98eeb98adb`
- bytes: 3,531,296
- SHA-256: `1e2dc9ffed295b4f2e3c555769971777ac8fb37bb0ad5e6dab506d8da0b42890`
