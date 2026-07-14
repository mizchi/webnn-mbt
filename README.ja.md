# webnn-mbt

[English](./README.md) | 日本語

MoonBit向けのWebNN backendおよびTFLite inference runtimeです。低レベルのWebNN binding、Chrome間のcompatibility adapter、型付きgraph builder、program cache、execution pool、TFLiteからWebNNへのloweringを提供します。Chrome Canaryのheadless modeで実装との互換性を継続検証しています。

現時点では `float32`、固定 shape、named multi-input/output の推論を対象にしています。対応演算は次のとおりです。

- `add` / `sub` / `mul` / `div`
- `reduce_mean` / `gather` / `slice` / `concat`
- `matmul`（rank 2以上、batch dimensionのbroadcast対応）
- `conv2d`（NCHW/OIHW と NHWC/HWIO）
- `max_pool2d` / `average_pool2d`
- `sigmoid` / `tanh` / `gelu`（WebNN仕様のexact erf定義）
- `layer_normalization`
- `clamp` / `relu`
- `softmax`
- `reshape`
- `transpose`

## 必要な環境

- MoonBit 0.10.3 以降
- Node.js 24 以降
- pnpm
- just
- Google Chrome Canary

macOS では Playwright の `chrome-canary` channel を利用します。テストは隔離された browser profile を使うため、通常の Canary で設定した `chrome://flags` には依存せず、次の feature flag を起動引数として渡します。

```text
--enable-features=WebMachineLearningNeuralNetwork
```

## 実行

```bash
just install
just check
```

個別のタスクも実行できます。

```bash
just unit
just build
just e2e
just fixture-bert
just bench
just bench-mnist
just bench-mnist-cache
just bench-mnist-pool
just bench-named-io
just bench-tiny-cnn
just bench-transformer
just bench-attention
just bench-encoder
just bench-encoder-stack
just bench-bert-encoder
just bench-litert
just bench-tflite
just bench-tflite-runner-cache
just bench-mobilenet-v2
```

## ワークスペース

[`moon.work`](./moon.work) により、公開ライブラリとブラウザ利用側を分離しています。

- `.` は `mizchi/webnn` ライブラリモジュールです。各パッケージは `src/` 以下に置きます。
- [`examples/playground`](./examples/playground) はリポジトリ内で利用する `mizchi/webnn-examples` アプリケーションモジュールです。ワークスペース内の `mizchi/webnn` に依存し、benchmark・MNIST fixture・Playwright から使う browser API を所有します。

アプリケーションは実装packageへ直接依存せず、整理されたroot facadeをimportできます。

```moonbit
import {
  "mizchi/webnn",
}
```

root packageは、検証済みのshape、LiteRT/TFLite、WebNN graph、program、cache、runtimeのコントラクトを再公開します。モデル構築に特化したpackageは`mizchi/webnn/model`、`mizchi/webnn/bert`、`mizchi/webnn/backend/cpu`として引き続き利用できます。

`just build` は playground アプリケーションを明示的にビルドします。MoonBit が生成する JavaScript entry point は `_build/js/release/build/mizchi/webnn-examples/app/app.js` で、[`public/index.html`](./public/index.html) から読み込みます。

## CI

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) は push、pull request、手動実行に加え、Chrome Canary の更新検知用に毎日 03:17 JST に動きます。権限はリポジトリ内容の読み取りだけに制限し、外部 Action は commit SHA で固定しています。

- `MoonBit contracts`: CI設定契約、`moon fmt --check`、`moon check`、117件のunit test、release build
- `WebNN Canary E2E`: Ubuntu 24.04上のPlaywright配布 `chromium-tip-of-tree`（Chrome Canary for Testing headless shell）で72件のWebNN E2E

失敗時はPlaywrightのHTML reportとtraceを7日間artifactとして保存します。CIと同じブラウザをローカルで再現する場合は次を実行します。

```bash
pnpm exec playwright install --only-shell chromium-tip-of-tree
WEBNN_BROWSER_CHANNEL=chromium-tip-of-tree just ci-e2e
```

MoonBitは公式配布がrolling channelのため、CIでは`latest`を使用して`moon version --all`を記録します。MoonBitの更新による非互換も日次CIで検出します。

ベンチ対象の shape、warmup、反復数も指定できます。

```bash
just bench 64,128,256 10 30
just bench-mnist 1,16,100 10 30
just bench-mnist-cache 1,16,100 10 30
just bench-mnist-pool 1,2,4,8 256 10 30
just bench-named-io 64,1024,16384 10 30
just bench-tiny-cnn 1,2,4,8,16,32,64 10 30 nchw,nhwc
just bench-transformer 32,64,128,256 16 10 30
just bench-attention 32,64,128,256 16 4 causal 10 30
just bench-encoder 32,64,128,256 2 16 4 causal-padding 10 30
just bench-encoder-stack 1,2,4 2 16 64 4 causal-padding 10 30
just bench-bert-encoder 1,2,4 2 16 64 4 padding 10 30
just bench-litert 10 30
just bench-tflite 10 30
just bench-tflite-runner-cache 10 30
just bench-mobilenet-v2 3 10
```

測定方法とローカル M5 での結果は [BENCHMARK.md](./BENCHMARK.md) に記録しています。

## 設計

モデル定義は MoonXi Net を参考にした tagless-final 形式です。`Linear`、`Mlp`、`TinyCnn`、`LayerNorm`、`FeedForward`、`SelfAttention`、`TransformerEncoderBlock/Stack`、`BertEncoderBlock/Stack`の`forward[T : TensorOps]`はbackendに依存せず、同じコードを`CpuTensor`と`WebNNTensor`で実行できます。`TransformerEncoderBlock`はpre-norm、`BertEncoderBlock`はHugging Face BERT checkpointと同じpost-normのresidual境界を定義します。既存の`TransformerBlock`は旧feed-forwardベンチとの互換用です。

```text
model/Linear
    |
    v
tensor/TensorOps
    |---------------------|
    v                     v
backend/cpu          backend/webnn
即時計算              MLOperand を構築
                          |
                          v
                    compile / dispatch / read
```

各層の責務は分離しています。

- `mizchi/webnn`: 主要な推論コントラクトをまとめた公開facade
- `shape`: shape 検証と演算後 shape の推論
- `tensor`: backend 共通の演算コントラクト
- `model`: backend 非依存のモデル定義
- `bert`: Hugging Face named tensorの検証、dense weight転置、BERT parameterへの変換
- `backend/cpu`: differential test 用の eager CPU 実装
- `webnn/raw`: WebNN JavaScript API の薄い FFI
- `webnn/compat`: 現行 Chrome の `deviceType` と最新仕様の `accelerated` の差を吸収
- `backend/webnn`: symbolic tensor、compile、program cache、execution pool、tensor lifecycle
- `examples/playground/src/app`: Playwright から呼び出す責務別の利用側 browser API adapter
- `examples/playground/src/benchmark`: CPU/WebNN の正しさを比較しながら setup と steady-state を計測
- `litert`: named tensor、float32 constant、演算列を表す source-neutral な IR と `TensorOps` lowerer
- `examples/playground/src/mnist`: 学習済みMLP weights・IDX loader・CPU/WebNN evaluation/benchmark

Transformer系の公開APIは、検証済みのconfig、backend非依存parameter、materialize済みmodelを分離します。

```text
SelfAttentionConfig / TransformerEncoderConfig / TransformerEncoderStackConfig
                         + Parameters (owned float32 arrays per layer)
                         + AttentionMask
                                     |
                    |----------------|----------------|
                    v                                 v
@cpu.materialize_transformer_encoder_stack   builder.materialize_transformer_encoder_stack
                    |                                 |
                    v                                 v
TransformerEncoderStack[CpuTensor]     TransformerEncoderStack[WebNNTensor]
```

`SelfAttentionConfig::new(width, heads)`、`TransformerEncoderConfig::new(width, heads, hidden_size, input_rank, epsilon)`、`TransformerEncoderStackConfig::new(layers, ...)`がdimensionと層数を検証します。`SelfAttentionParameters`、`FeedForwardParameters`、`TransformerEncoderParameters`、`TransformerEncoderStackParameters`は入力配列をcopyして所有し、getterもcopyを返します。CPU/WebNNの`materialize_transformer_encoder_stack()`はmask constantを一度だけmaterializeし、全層で共有します。

`AttentionMask`は全batch共通の`causal` / `padding` / `causal_padding`に加え、`batched_padding` / `batched_causal_padding`で`[batch,1,tokens,tokens]`を生成します。`additive(shape, values)`では検証済みの任意additive maskを渡せます。

### BERT encoder loader

`BertEncoderConfig`と`BertEncoderParameters`はpost-norm BERT専用の契約です。`@bert.NamedTensor`でHugging Face/PyTorch checkpointのtensor名・shape・float32値を渡し、`load_hugging_face_encoder(config, tensors, "bert.encoder")`でparameterへ変換します。`query/key/value`、attention output、intermediate、output denseのweightはcheckpointの`[out_features,in_features]`から`Linear`の`[in_features,out_features]`へ転置します。欠落、重複、shape不一致は完全なtensor名を含む`TensorError`になります。embeddingsやpoolerなどencoder外のtensorは無視します。

`parse_safetensors(bytes)`はSafeTensorsの8-byte little-endian header長、UTF-8 JSON header、data section相対offset、row-major little-endian F32を検証して`Array[NamedTensor]`を返します。offsetの穴・重なり・末尾の未参照byte、shapeとbyte長の不一致も拒否します。`just fixture-bert`は2層・width 8・32 tensorのdeterministic `.safetensors`を生成し、Canary E2Eで`fetch → parse → mapping → CPU/WebNN`を検証します。

CPU/WebNNの`materialize_bert_encoder_stack()`はpadding maskを一度だけmaterializeして全層で共有します。推論用のためdropoutは含みません。

`WebNNProgramCache` は compiled graph と prepared input/output tensor をまとめた
`WebNNProgram` を内部所有します。公開APIは `run_or_compile()` と
`run_named_or_compile()` のみで、呼び出し元へprogramを返さないため、cached programを
誤って `destroy()` する経路を作りません。cache key は versioned canonical string で、graph identity、
device preference、順序付き input shapes、execution pool sizeを含みます。MNIST の graph identity には
MLP topology/layout のversionとfixture weightsのSHA-256を含め、weightsやshapeが
変わったprogramを誤って再利用しないようにしています。cache missの途中で同じkeyが
登録された場合は、後から完成したprogramを破棄して既存entryを採用します。cached resourceの解放は
`clear()` に一元化されています。`clear()` はgenerationを進めるため、compile中だった古いentryは
呼び出し結果だけ返した後に破棄され、clear後のcacheへ再登録されません。

`WebNNProgram` はpool size分の独立したinput/output tensor slotを持ち、FIFO schedulerが
空きslotへrunを割り当てます。pool size 1では同じprepared tensorへのwrite/dispatch/readを
直列化し、2以上ではslot間の並行実行を許可します。途中のrunが失敗しても後続runを継続し、
`destroy()` は新規runを拒否してaccepted済みrunの完了後に全slotとcontextを解放します。

既定pool sizeは逐次latencyとmemoryを優先して1です。今回のMNIST同時request測定では2で
throughput改善の大半を得られたため、並行用途では2を最初の候補とし、4以上は実測して
選びます。

graphのI/Oは`WebNNInput`、`WebNNOutput`、runtime値は`WebNNNamedValues`で表現します。
`compile_named()` / `run_named()` は配列順ではなく名前で入力を照合し、出力はgraph宣言順で
返します。空名、重複名、未知名、不足、要素数不一致はWebNN dispatch前にMoonBit側で拒否します。
従来の`compile_single()` / `run()`は1入力・1出力用の互換wrapperとして残しています。

WebNN context の `opSupportLimits()` を読み、対象 operator が報告されていない場合は graph 構築前にエラーにします。shape と input/constant の要素数も MoonBit 側で検証します。`probe_capabilities(preference)` は `WebNNCapabilities` として、認識した context contract、preferred input layout、`opSupportLimits()`、prepared tensor I/O、backendが使うoperator群を返します。Canary E2Eはこの値を検証し、仕様または実装の更新を検出します。

`WebNNInput` / `WebNNOutput` が名前・shape・operandを一体で保持するため、compile時に別builderのoperandを誤って指定できません。`WebNNGraphBuilder`はcontextの所有者です。単発sessionは`defer graph.destroy()`、compiled programは`program.destroy()`で全slotとcontextを解放します。

`LiteRtModel` は input・constant・intermediate を名前とshapeで宣言し、演算列を検証してから
generic `TensorOps` へ lower します。現在は `add`、`sub`、`mul`、`div`、`reduce_mean`、`gather`、`slice`、`gelu`、`layer_normalization`、`concat`、`matmul`、`conv2d`、`max_pool2d`、`average_pool2d`、`sigmoid`、`tanh`、`clamp`、`relu`、`softmax`、`reshape`、`transpose` を表現できます。`gelu` は現行WebNN仕様のexact `0.5*x*(1+erf(x/sqrt(2)))`をCPU/WebNN共通の契約にします。`layer_normalization(input, scale, bias, axes, epsilon)` は非空かつ重複のない明示的なaxisを受け、`scale`と`bias`にはaxis順のdimensionからなるshape、`epsilon`には正の値を要求します。WebNN 固有の `WebNNGraphBuilder.lower_litert(model)` が input/constant を materialize し、`compile_litert_program(model)` / `compile_litert_program_pool(model, pool_size)` が lower・named I/O binding・compile・prepared execution をまとめて所有します。

`TfliteModel::parse(bytes)` は [TFLite schema](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/3e5424592ce7a5eeded530a58cc42f9fb981e40a/tensorflow/lite/schema/schema.fbs) の FlatBuffer を読み、単一subgraphの `ADD`、`SUB`、`MUL`、`DIV`、`MEAN`、`GATHER`、`SLICE`、`CONCATENATION`、`RELU`、`RELU_N1_TO_1`、`RELU6`、`LOGISTIC`、`TANH`、`SOFTMAX`、`FULLY_CONNECTED`、`CONV_2D`、`DEPTHWISE_CONV_2D`、`AVERAGE_POOL_2D`、`MAX_POOL_2D`、`RESHAPE`、`TRANSPOSE` をfloat32 IRへ変換します。`MEAN` はconstant INT32 axes、`GATHER` はconstant INT32 indices、`SLICE` はconstant INT32 begin/size（`-1` は末尾まで）のみを受け、axisと出力shapeを検証します。`CONCATENATION` は2個以上のinputをbinary `concat`列へlowerし、negative axisとfused activationを扱います。`UINT8`/`INT8` constantは `scale * (q - zero_point)` で静的dequantizeし、per-axis metadata、current schemaの`quantized_dimension`と旧encoderのfield位置の差、量子化`INT32` biasも扱います。rank 1 bias がproducerのNHWC channel axisを保持する実モデルは、scale数がbias要素数と一致する場合にaxis 0へ正規化します。operator出力はbufferの有無でなくgraph topologyからintermediateと判定します。rank 2超の`FULLY_CONNECTED` inputは `[batch, flattened]` を明示的に`reshape`してからlowerします。custom operatorは実行前に名前付きで拒否します。

任意の .tflite bytes を実行する用途には低レベルの`TfliteRunner`/`TfliteRunnerCache`に加え、`WebNNRuntime` を使えます。`WebNNRuntime::new(preference)` または `new_with_pool(preference, pool_size)` は、各32モデル・source FlatBuffer合計64 MiB上限のWebNN LRU cacheを内部所有します。上限・fallbackを明示するには`new_with_options(preference, pool_size, capacity, fallback)`、WebNN失敗時のCPU再試行を選ぶには`new_with_cpu_fallback(preference, pool_size, capacity)`を使います。entry数とbyte budgetを個別に制御するには`new_with_cache_limits(preference, pool_size, webnn_entries, webnn_bytes, cpu_entries, cpu_bytes, fallback)`を使います。byte budgetはブラウザがcompiled WebNN resourceの実メモリ量を公開しないため、正確に追跡できるモデルFlatBuffer bytesで管理します。単体cacheにも`TfliteRunnerCache::new_with_limits(entries, bytes)`と`TfliteCpuRunnerCache::new_with_limits(entries, bytes)`があります。予算より大きいmodelは既存entryをすべて追い出さず、transientに実行して直後に解放します。

`metrics()`はWebNN cacheの累積hit/miss/eviction/entry数・resident bytesと、CPU fallback数・CPU parsed-model cacheの同種metricsを返します。`clear()`はWebNN programとCPU parsed modelを解放してもcounterをリセットせず、実行中compileはgenerationで無効化するため、clear後に完了してもcacheへ再挿入されません。大きなmodelの定常実行では、`TfliteModelArtifact::from_bytes(bytes)`でbytesを所有しdigestを一度だけ計算し、`run_prepared_tflite(artifact, inputs)`を使うと毎回のSHA-256を省けます。CPU fallbackは同じquantized I/O contractの`TfliteCpuRunner`を使い、初回にparse/lowerしたmodelをdigest keyで再利用します。ただしWebNNを毎回試みるため、これは通常経路の性能代替ではなく可用性のためのポリシーです。run は TFLite の tensor name で input を照合し、float32 tensor には `TfliteRunnerInput::float32`、量子化 tensor には `::quantized` を受け付けます。出力は元の TFLite dtype に従い float32 または raw `Array[Int]` です。

## 実 TFLite fixture

`just fixture-tflite` はTensorFlow Lite Microのcommitを固定し、SHA-256を検証してfixtureを取得します。

- `micro_speech_quantized.tflite`（18,800 bytes）は `RESHAPE → DEPTHWISE_CONV_2D → FULLY_CONNECTED → SOFTMAX` の量子化実モデルです。raw INT8 input/output、INT32 bias、rank 4からのflatten、およびruntime cacheをCanary E2Eで検証します。
- `person_detect.tflite`（300,568 bytes）は 96×96 grayscale の量子化人物検出CNNです。rank 1のper-channel INT32 biasがproducerのaxis 3を保持する実モデル表現を正規化し、zero inputのraw INT8 I/O・compiled runtime cacheをCanary E2Eで検証します。
- `mobilenet_v2_1.0_224_inat_bird_quant.tflite`（3,531,296 bytes）は Coral Edge TPU の量子化MobileNet V2分類モデルです。delegateを使わず、parse/lower/compile、raw UINT8 I/O、cache hitをCanary E2Eで検証します。
- `audio_preprocessor_int8.tflite`（8,772 bytes）は `SignalWindow` custom operatorを含むnegative fixtureです。現時点では `unsupported TFLite custom operator: SignalWindow` と拒否されることを固定しています。

## ブラウザ API

ビルドされたページは `globalThis.webnnPlayground` に検証用 API を公開します。

- `compatibilityMode()`
- `supportedOperators()`
- `probeWebnnCapabilities()`
- `runCpuLinear()`
- `runAdd()`
- `runConcat()`
- `runLinear()`
- `runLiteRtLinear()`
- `runTfliteAdd()`
- `runTfliteSub()`
- `runTfliteAddRelu6()`
- `runTfliteFullyConnected()`
- `runTfliteFullyConnectedRelu6()`
- `runTfliteConv2d()`
- `runTfliteReshape()`
- `runTfliteTranspose()`
- `runTfliteDepthwiseConv2d()`
- `runTfliteAveragePool2d()`
- `runTfliteMaxPool2d()`
- `runTfliteLogistic()`
- `runTfliteTanh()`
- `runTfliteQuantizedAdd()`
- `runTfliteQuantizedAddRaw()`
- `runTflitePerAxisInt8Raw()`
- `runTfliteReluN1To1()`
- `runTfliteRelu6()`
- `benchmarkLiteRtLinear(warmup, iterations)`
- `benchmarkTfliteFullyConnected(warmup, iterations)`
- `benchmarkTfliteSub(warmup, iterations)`
- `benchmarkTfliteDepthwiseConv2d(warmup, iterations)`
- `benchmarkTfliteMaxPool2d(warmup, iterations)`
- `benchmarkTflitePerAxisInt8Raw(warmup, iterations)`
- `runTfliteRunnerCachePerAxisInt8Raw()`
- `runTfliteRuntimeCachePerAxisInt8Raw()`
- `runTfliteCpuPerAxisInt8Raw()`
- `runTfliteRuntimeCpuFallbackPerAxisInt8Raw()`
- `probeTfliteRuntimePolicy()`
- `runMicroSpeechFixture()`
- `runPersonDetectionFixture()`
- `probeMobileNetV2Fixture()`
- `runMobileNetV2Fixture()`
- `benchmarkMobileNetV2Fixture(warmup, iterations)`
- `probeAudioPreprocessorFixture()`
- `probeTfliteRunnerCacheConcurrency()`
- `runTfliteRunnerNamedAdd()`
- `runTfliteRunnerTwoInputTwoOutput()`
- `benchmarkTfliteRunnerCachePerAxisInt8Raw(warmup, iterations)`
- `runTransformPipeline()`
- `benchmarkMatmul(size, warmup, iterations)`
- `benchmarkTransformerBlock(tokens, width, warmup, iterations)`
- `benchmarkSelfAttention(tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkTransformerEncoder(batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkTransformerEncoderStack(layers, batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `benchmarkBertEncoder(layers, batchSize, tokens, width, heads, maskMode, warmup, iterations)`
- `runBertSafetensorsFixture()`
- `evaluateMnist(count)`
- `benchmarkMnist(batchSize, warmup, iterations)`
- `clearMnistCache()`
- `runMnistCached(batchSize)`
- `benchmarkMnistCache(batchSize, warmup, iterations)`
- `probeConcurrentProgram()`（並行run・deferred destroyのE2E検証用）
- `probeExecutionPool()`（複数slot割り当てのE2E検証用）
- `benchmarkMnistPool(poolSize, requestCount, warmup, iterations)`
- `runNamedMultiIo()`
- `benchmarkNamedIo(size, warmup, iterations)`
- `runConvSoftmax()`
- `benchmarkTinyCnn(batchSize, warmup, iterations)`
- `benchmarkTinyCnnLayout(inputLayout, batchSize, warmup, iterations)`
- `runPreferredLayoutConv()`

`runLinear()` と `runCpuLinear()` は同じ generic `Linear.forward` を使い、Playwright で結果を比較します。

## 制限

- WebNN の全 operator、全 data type、動的 shape には未対応です。
- `.tflite` parserは単一subgraph・`ADD`/`SUB`/`MUL`/`DIV`/`MEAN`/`GATHER`/`SLICE`/`CONCATENATION`/`RELU`/`RELU_N1_TO_1`/`RELU6`/`LOGISTIC`/`TANH`/`SOFTMAX`/`FULLY_CONNECTED`/`CONV_2D`/`DEPTHWISE_CONV_2D`/`AVERAGE_POOL_2D`/`MAX_POOL_2D`/`RESHAPE`/`TRANSPOSE` に限定しています。`GELU`と`layer_normalization`はgeneric IR/APIでは利用できますが、TFLite parserからはまだ生成しません。`CONCATENATION` は負のaxis、2個以上のinput、fused activationを扱います。`UINT8`/`INT8` はpositive scale と zero-pointを必須とし、constantはscalar per-tensorまたはshapeに適合するper-axis quantizationでfloat32へ静的dequantizeします。量子化`INT32`はbias constantのみ対応し、int32の演算・runtime入力は未対応です。fused activation は `NONE`、`RELU`、`RELU_N1_TO_1`、`RELU6`、`TANH` に限り、`SIGN_BIT` は未対応です。`INT16`、`FLOAT16` は未対応です。複数subgraph/control-flow、custom operator（例: `SignalWindow`）は未対応です。
- `conv2d`はNCHW input/OIHW filter と NHWC input/HWIO filterをサポートします。padding、stride、dilation、groupsに対応し、biasはレイアウトに合わせてNCHWでは`[1,C,1,1]`、NHWCでは`[1,1,1,C]`の`add`で表現します。`WebNNGraphBuilder.preferred_input_layout()` はcontextの`opSupportLimits().preferredInputLayout`を読んで選択に利用できます。
- `accelerated` contract では CPU/accelerator の区別しか指定できないため、`Npu` は `accelerated: true` に変換されます。
- 現行 Canary の `deviceType: "npu"` が成功しても、実際に Apple Neural Engine が選択されたことまでは WebNN API から確認できません。
- native backend はこの実装には含めず、将来 Core ML / ONNX Runtime backend として追加する想定です。
- BERT loaderはencoder layerのみを対象とし、embeddings、pooler、task head、ONNX、`gelu_new`には未対応です。SafeTensors parserは単一fileのF32・正の固定shapeのみを受け、F16/BF16、sharded index、zero-sized tensorには未対応です。
- execution poolは同じbrowser threadとWebNN contextを共有します。pool sizeを増やしてもhardware/backend内部の並列度以上には速くならず、slotごとのtensor memoryは増えます。
- Canary 152では同じ`MLOperand`を2つのoutput名へ直接指定するとbuildが`Context is lost`で失敗しました。別operand化する`reshape`で回避できます。

参考:

- [WebNN specification](https://www.w3.org/TR/webnn/)
- [SafeTensors format](https://github.com/huggingface/safetensors#format)
- [MoonXi Net](https://www.moonbitlang.com/pearls/moonxi-net)
- [Playwright browser channels](https://playwright.dev/docs/browsers)
