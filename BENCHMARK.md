# Benchmark

## 測定方法

`just bench` は Chrome Canary を headless で起動し、正方 `float32` matmul を次の経路で比較します。

- CPU: MoonBit JS backend の `CpuTensor.matmul`
- WebNN fresh: graph は再利用し、input/output tensor は推論ごとに確保・破棄
- WebNN reuse: graph と input/output tensor の両方を再利用

setup は次の3区間に分けます。

- context: `navigator.ml.createContext()`
- compile: `MLGraphBuilder.build()`
- prepare: input/output `MLTensor` の作成

steady-state の WebNN サンプルには input write、dispatch、readback、`ArrayBuffer` から MoonBit `Array[Float]` への変換が含まれます。小さい shape が `performance.now()` の分解能で 0 ms にならないよう、複数推論を1サンプルにまとめ、1推論あたりの時間へ正規化しています。

CPU 実装は比較・正しさ確認用の素朴な O(n³) JS 実装であり、BLAS や native CPU backend との比較ではありません。全サンプルで CPU/WebNN の最大絶対誤差も検証します。

## 2026-07-11

- Apple M5、10 cores、32 GB memory
- Chrome Canary 152.0.7942.0
- headless
- WebNN `deviceType: "npu"`
- warmup: 10
- samples: 30

```text
| size | ops/sample | CPU p50 ms | WebNN fresh p50 ms | WebNN reuse p50 ms | reuse p95 ms | speedup | context ms | compile ms | prepare ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 20 | 0.060 | 0.095 | 0.050 | 0.065 | 1.20x | 1.400 | 1.300 | 0.100 | 0.00e+0 |
| 64 | 10 | 0.510 | 0.140 | 0.120 | 0.160 | 4.25x | 0.400 | 1.000 | 0.000 | 0.00e+0 |
| 128 | 3 | 4.933 | 0.367 | 0.300 | 0.400 | 16.44x | 0.400 | 3.900 | 0.100 | 0.00e+0 |
| 256 | 1 | 45.400 | 1.300 | 1.200 | 2.100 | 37.83x | 1.400 | 4.100 | 0.000 | 0.00e+0 |
```

setup は各 shape につき1回だけの測定なので、steady-state の p50/p95 より変動しやすい値です。また、`deviceType: "npu"` は選択希望であり、Apple Neural Engine が実際に使われたことを WebNN API から確認できるわけではありません。

## MNIST MLP

`just bench-mnist` は学習済み `784 → 128 → 10` MLPを実行します。モデルのforwardはCPU/WebNNで共通で、`Linear → ReLU → Linear` です。fixture load、context作成、weight constantを含むgraph構築、compile、tensor prepareを別々に測定します。

```text
| batch | ops/sample | CPU p50 ms | WebNN fresh p50 ms | WebNN reuse p50 ms | reuse p95 ms | speedup | fixture ms | context ms | graph ms | compile ms | prepare ms | max error | correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.240 | 0.120 | 0.100 | 0.140 | 2.40x | 7.500 | 1.300 | 0.600 | 1.600 | 0.200 | 5.25e-6 | 1/1 |
| 16 | 1 | 3.900 | 0.100 | 0.100 | 0.200 | 39.00x | 4.500 | 0.300 | 0.300 | 1.200 | 0.000 | 3.81e-6 | 15/16 |
| 100 | 1 | 25.400 | 0.300 | 0.300 | 0.400 | 84.67x | 3.600 | 0.400 | 0.300 | 1.200 | 0.100 | 3.81e-6 | 96/100 |
```

CPU/WebNNの予測classは100件すべて一致し、双方とも96/100正解でした。fixture loadは両backendに共通するためspeedupには含めていません。

batch 1ではWebNN setupが約3.7 msあり、steady-stateの差から約27回の再利用で償却します。batch 16以上では、今回の素朴なCPU baselineに対してはsetupを含む初回推論からWebNNが優位です。

## 実装への反映

tensor 再利用による改善は 32×32 で約 47%、64×64 で約 14%、128×128 で約 18%、256×256 で約 8% でした。この結果を受け、繰り返し推論用の `WebNNExecution` は input/output tensor を保持して再利用します。`WebNNSession.run()` は一度だけ実行するための簡便 API として残しています。

小さい matmul の一度限りの実行では setup cost が計算時間を上回ります。backend の自動選択を追加する場合は、shape だけでなく予定反復回数も判断材料にする必要があります。

実モデルでも同じ傾向が確認できたため、次の性能施策はoperator単体の最適化より、compiled graph/sessionをgraph hashとshapeでcacheしてsetupを償却することです。

## Compiled program cache

`just bench-mnist-cache` は fixture load と CPU reference 計算を測定外に置き、
次の2経路を比較します。

- cold miss: cache clear後の`run_or_compile()`、context作成、graph構築、compile、tensor prepare、1回の推論
- cache hit: `graph identity + device + input shapes` のkey解決と、prepared tensorを再利用した推論

cold/hitとも30サンプルを取り、小さいbatchのhitは複数推論を束ねて1推論あたりへ
正規化しています。cold測定ではサンプルごとにentryを破棄するため、毎回必ずmissに
なっていることもreportの`coldMisses`で検証します。

```text
| batch | ops/sample | cold p50 ms | cold p95 ms | hit p50 ms | hit p95 ms | cold/hit | cache entries | max error | matches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 1.300 | 3.500 | 0.060 | 0.100 | 21.67x | 1 | 5.25e-6 | 1/1 |
| 16 | 1 | 1.300 | 6.800 | 0.100 | 0.200 | 13.00x | 1 | 3.81e-6 | 16/16 |
| 100 | 1 | 5.000 | 10.600 | 0.400 | 0.500 | 12.50x | 1 | 3.81e-6 | 100/100 |
```

cache hit後はcontext/compile/prepareを推論経路から除外できました。cold p95はrun間の
変動が大きいため、cold startを1回だけ測ると結果を過大・過小評価しやすい点にも注意が
必要です。cache entryは明示的なclearでprepared tensorを先に、contextを後に破棄します。

### Concurrent run serialization

cache導入後、同一programのprepared input/output tensorへ並行runが入ると、`destroy()`や
別runのwriteと進行中readが競合する問題が残っていました。`WebNNProgram`にFIFO Promise
queueを追加し、runを直列化、destroyをqueue末尾へ遅延しました。異なる2入力の
`Promise.all`、失敗run後の回復、destroy後のrun拒否をCanary E2Eで検証しています。

generation-aware cache追加後のcache hit p50はbatch 1/16/100でそれぞれ
`0.06 / 0.10 / 0.40 ms` です。cold startの変動に比べて定常経路の追加オーバーヘッドは
観測されませんでした。

## Execution pool

`just bench-mnist-pool` は同じcompiled graphに複数のprepared input/output tensor slotを
作り、batch 1のMNIST推論を256件同時投入します。各サンプルは256件すべてが完了するまでの
group latencyです。warmup 10回、30サンプルで測定しました。

```text
| pool | max concurrency | setup ms | group p50 ms | group p95 ms | ms/request | requests/s | vs pool 1 | max error | matches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 3.200 | 14.000 | 16.800 | 0.055 | 18285.7 | 1.00x | 5.25e-6 | 256/256 |
| 2 | 2 | 1.500 | 11.300 | 14.100 | 0.044 | 22654.9 | 1.24x | 5.25e-6 | 256/256 |
| 4 | 4 | 1.700 | 11.000 | 14.300 | 0.043 | 23272.7 | 1.27x | 5.25e-6 | 256/256 |
| 8 | 8 | 1.900 | 11.000 | 13.000 | 0.043 | 23272.7 | 1.27x | 5.25e-6 | 256/256 |
```

逆順（8, 4, 2, 1）でもpool 2/4/8はgroup p50 `10.1–10.3 ms`、pool 1は
`13.4 ms`で、2 slotが改善の大半を占める傾向を再確認しました。setup値は最初に作るcontextの
cold startに左右されるため、pool間の判断にはsteady-state group latencyを使っています。

既定値は逐次推論とmemory消費を優先して1を維持します。同時requestを処理する用途では
pool 2を開始点とし、4以上はmodel shape、request burst、対象hardwareごとにbenchmarkして
選びます。pool sizeはprepared tensor数を変えるためprogram cache key v2にも含めています。

## Named multi-I/O

`just bench-named-io` は2つのdynamic inputを加算し、同じ計算結果を1本または2本のnamed
outputとしてreadbackします。2本目は同じoperandの直接再公開ではなく、同shapeの`reshape`で
別operand化しています。これはCanary 152で同一operandを2つのoutput名へ指定するとbuildが
`Context is lost`になるためです。

```text
| values | ops/sample | 1 output p50 ms | p95 ms | 2 outputs p50 ms | p95 ms | 2/1 | delta ms | compile 1/2 ms | prepare 1/2 ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 20 | 0.030 | 0.045 | 0.040 | 0.055 | 1.33x | 0.010 | 1.300/1.300 | 0.300/0.000 | 0.00e+0 |
| 1024 | 5 | 0.040 | 0.100 | 0.080 | 0.100 | 2.00x | 0.040 | 0.900/1.000 | 0.200/0.100 | 0.00e+0 |
| 16384 | 5 | 0.260 | 0.280 | 0.460 | 0.540 | 1.77x | 0.200 | 0.900/0.900 | 0.100/0.200 | 0.00e+0 |
```

2本目のoutput readback追加コストは64要素で0.01 ms、1024要素で0.04 ms、16384要素で
0.20 msでした。output数だけでなく転送するbyte数が効くため、不要な中間tensorをgraph
outputへ公開しないことが重要です。すべてのshapeでCPU計算との最大絶対誤差は0でした。

## LiteRT IR lowering

`just bench-litert` は、LiteRT/TFLite の binary parser より前段の source-neutral IR を
WebNNへ lower する最小の `input → Matmul → Add` graph を測定します。`lower` にはIR構築、
shape/dataflow検証、`MLOperand`作成が含まれます。compileとprepared tensor生成は別計測です。

```text
| ops/sample | context ms | lower ms | compile ms | prepare ms | run p50 ms | run p95 ms | output |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 1.000 | 0.900 | 1.500 | 0.100 | 0.030 | 0.040 | [8, 11] |
```

この小さなgraphではlower costはcompile costと同程度で、定常実行より大きくなります。これは
one-shotでの自動lowerを選ぶ根拠にはならず、model IR とcompiled programをcacheする必要がある
ことを示します。なお、この数値はFlatBufferのparseや量子化constantのdequantizeを含みません。

## TFLite FlatBuffer parsing

`just bench-tflite` は float32 の単一subgraphで表現した `FULLY_CONNECTED`、`SUB`、
`DEPTHWISE_CONV_2D`、`MAX_POOL_2D` と、per-axis `INT8` runtime I/O を持つ `ADD` を parse します。`SUB` はnamed 2入力を WebNN `sub` へ lowerします。前者は TFLiteの `[output, input]` weight を
`matmul` 向けに転置してbias addへ、depthwise は`[1,H,W,C×multiplier]` filterを grouped NHWC/HWIO
`conv2d` とbias addへ、pooling は WebNN `maxPool2d` へ lowerします。parseは FlatBufferのtable/vtable
参照、tensor/buffer/operatorの検証、IR構築を含みます。`INT8_PER_AXIS_RAW` はrank-2 `[1,2]` tensorの
axis 1 に沿ったscale/zero-pointを使い、各定常サンプルにraw inputのdequantizeとoutputのrequantizeを含めます。

以下は Chrome Canary 152.0.7946.0、headless、`deviceType: "npu"`、warmup 10回・30 samplesの再計測です。

```text
| model | ops/sample | parse ms | context ms | lower ms | compile ms | prepare ms | run p50 ms | run p95 ms | output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FULLY_CONNECTED | 20 | 1.800 | 1.000 | 0.400 | 1.300 | 0.200 | 0.030 | 0.040 | [8, 11] |
| SUB | 20 | 0.200 | 0.600 | 0.000 | 1.000 | 0.200 | 0.030 | 0.040 | [2, 10] |
| DEPTHWISE_CONV_2D | 20 | 0.500 | 0.500 | 0.100 | 1.000 | 0.200 | 0.035 | 0.050 | [63, 730, 79, 930, 111, 1330, 127, 1530] |
| MAX_POOL_2D | 20 | 0.400 | 0.500 | 0.000 | 0.900 | 0.100 | 0.035 | 0.045 | [5, 6, 8, 9] |
| INT8_PER_AXIS_RAW | 20 | 0.400 | 0.500 | 0.000 | 1.000 | 0.000 | 0.030 | 0.035 | [2, 3] |
```

過去の反復ではFULLY_CONNECTEDのcompileが `98.9 / 60.9 ms` となる外れ値も観測しました。初回のWebNN compileには大きな変動があり得るため、setup値は反復測定し、steady-stateのp50/p95とは分けて判断します。

この小modelではparse、context、compileはいずれも定常実行より大きいため、実運用ではmodel bytes
からのparse結果とcompiled programをともにcacheする必要があります。`ADD`、`SUB`、`MUL`、`CONCATENATION`、`RELU`、
`SOFTMAX`、`CONV_2D`、`AVERAGE_POOL_2D`、`RESHAPE`、`TRANSPOSE` もparser対応済みです。
`RESHAPE`/`TRANSPOSE` はconstant int32 の制御tensorを読みます。`UINT8`/`INT8` constantと
runtime I/O metadataはscalar per-tensorおよびshapeに適合するper-axisに対応済みですが、複数subgraph、
control-flow、custom operatorは測定対象外かつ未対応です。

## TfliteRunner cache

just bench-tflite-runner-cache は per-axis INT8 runtime I/O を持つ同じ ADD
FlatBuffer を繰り返し実行します。cold sample は cache clear 後の SHA-256、parse、
context 作成、compile、prepared tensor 作成、1回の推論を含みます。hit sample は
同じ SHA-256 の lookup、raw input の dequantize、dispatch、raw output の
requantize を含み、20回を束ねて1回あたりに正規化します。

Chrome Canary 152.0.7945.0、headless、WebNN npu、warmup 10回・30 samples:

| model | cold p50 ms | cold p95 ms | hit p50 ms | hit p95 ms | cold/hit | cache entries | raw output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8_PER_AXIS_RAW | 0.800 | 1.500 | 0.035 | 0.045 | 22.857x | 1 | [5, 10] |

この小モデルでも cache hit は cold p50 の約23分の1でした。bytes digest は毎回の
cache lookup に含まれるため、巨大モデルでは SHA-256 のコストも同じ測定で確認する必要があります。

### Runtime cache policy regression

LRU容量制限とmetrics追加後は `just bench-tflite-runner-cache 5 15` を再実行しました。
Chrome Canary 152.0.7946.0、headless、WebNN npuで、cold p50は`1.000 ms`、hit p50は
`0.045 ms`（22.22x）、hit p95は`0.080 ms`でした。LRU bookkeepingとcounter更新を
入れても、定常pathは0.1 ms未満です。容量超過・LRU順序・CPU fallbackは別途Canary E2Eで
固定し、fallbackのparse/lowerコストはこの通常経路のベンチには混ぜません。

## MobileNet V2 real-model cache

`just bench-mobilenet-v2` は Coral Edge TPU repositoryで配布される3.5 MBの量子化
MobileNet V2を、delegateなしで WebNN にlowerします。fixture fetchは測定外です。cold
sampleにはSHA-256、FlatBuffer parse、量子化constantのdequantize、context、lower、compile、
prepared tensor作成、1回の推論を含めます。bytes hit sampleには同じSHA-256 lookup、raw
UINT8入力のdequantize、dispatch、raw UINT8出力のrequantizeを含めます。prepared hit sampleは
`TfliteModelArtifact`の事前計算済みdigestを使い、同じ推論経路からSHA-256だけを除きます。

Chrome Canary 152.0.7946.0、headless、WebNN npu、warmup 2回・5 samples:

| model | cold p50 ms | bytes hit p50/p95 ms | prepared hit p50/p95 ms | cold/prepared | cache entries | output values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 UINT8 | 82.500 | 5.700 / 6.000 | 4.500 / 4.800 | 18.333x | 1 | 965 |

実モデルでもcompileを含む初回はprepared hitより約18倍遅くなります。`TfliteModelArtifact`は
3.5 MB bytesを一度コピーしてdigestを固定し、prepared hit p50を`5.7 ms`から`4.5 ms`へ約21%短縮
しました。定常経路には引き続きinput/outputの量子化変換とdispatch/readbackが残るため、artifactは
モデルload時に繰り返し推論する用途にだけ使うのが適切です。

## Tiny CNN

`just bench-tiny-cnn`は決定的なweightを持つ次のmodelを、同じgeneric `TinyCnn.forward`で
CPU/WebNNへ構築します。既定ではNCHW/OIHWとNHWC/HWIOの両方を測定します。NHWC時はconv filterと
flatten後のLinear weightを並べ替えるため、NCHWと同じ論理モデル・出力になります。

```text
NCHW [batch, 1, 8, 8]
  → Conv2d OIHW [4, 1, 3, 3]
NHWC [batch, 8, 8, 1]
  → Conv2d HWIO [3, 3, 1, 4]
  → bias add → ReLU
  → flatten [batch, 144]
  → Linear [144, 10]
  → Softmax(axis=1)
```

```text
NCHW/OIHW:

| batch | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error | probability sum error | matches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.020 | 0.040 | 0.080 | 0.50x | 1.300 | 0.300 | 2.000 | 0.200 | 7.45e-9 | 1.19e-7 | 1/1 |
| 2 | 3 | 0.033 | 0.067 | 0.100 | 0.50x | 0.200 | 0.100 | 1.000 | 0.000 | 1.49e-8 | 1.19e-7 | 2/2 |
| 4 | 3 | 0.067 | 0.067 | 0.100 | 1.00x | 0.200 | 0.100 | 1.000 | 0.100 | 1.49e-8 | 1.19e-7 | 4/4 |
| 8 | 1 | 0.100 | 0.100 | 0.100 | 1.00x | 0.200 | 0.100 | 1.400 | 0.100 | 1.49e-8 | 1.19e-7 | 8/8 |
| 16 | 1 | 0.300 | 0.100 | 0.200 | 3.00x | 0.300 | 0.000 | 1.100 | 0.000 | 1.49e-8 | 1.19e-7 | 16/16 |
| 32 | 1 | 0.500 | 0.100 | 0.200 | 5.00x | 0.300 | 0.100 | 1.000 | 0.000 | 1.49e-8 | 1.19e-7 | 32/32 |
| 64 | 1 | 1.100 | 0.100 | 0.200 | 11.00x | 0.300 | 0.100 | 0.900 | 0.100 | 1.49e-8 | 1.19e-7 | 64/64 |

NHWC/HWIO:

| batch | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error | probability sum error | matches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.020 | 0.040 | 0.080 | 0.50x | 0.200 | 0.200 | 1.000 | 0.000 | 7.45e-9 | 1.19e-7 | 1/1 |
| 2 | 3 | 0.033 | 0.033 | 0.100 | 1.00x | 0.300 | 0.000 | 1.000 | 0.100 | 1.49e-8 | 1.19e-7 | 2/2 |
| 4 | 3 | 0.067 | 0.067 | 0.100 | 1.00x | 0.200 | 0.100 | 0.900 | 0.100 | 1.49e-8 | 1.19e-7 | 4/4 |
| 8 | 1 | 0.100 | 0.100 | 0.100 | 1.00x | 0.200 | 0.000 | 1.000 | 0.000 | 1.49e-8 | 1.19e-7 | 8/8 |
| 16 | 1 | 0.300 | 0.100 | 0.100 | 3.00x | 0.300 | 0.000 | 1.000 | 0.000 | 1.49e-8 | 1.19e-7 | 16/16 |
| 32 | 1 | 0.500 | 0.100 | 0.200 | 5.00x | 0.300 | 0.000 | 1.000 | 0.000 | 1.49e-8 | 1.19e-7 | 32/32 |
| 64 | 1 | 1.100 | 0.100 | 0.200 | 11.00x | 0.300 | 0.000 | 1.100 | 0.200 | 1.49e-8 | 1.19e-7 | 64/64 |
```

このCanary contextの`preferredInputLayout`は`nhwc`でした。今回の定常 p50は両layoutで同等で、
batch 2だけNHWCが`0.033 ms`、NCHWが`0.067 ms`となりましたが、`performance.now()`の分解能に
近い値なので優劣の根拠にはしません。backendの既定は後方互換のNCHW/OIHWのままにし、モデルごとに
`preferred_input_layout()`を選べるようにしています。全batch・両layoutで予測classは一致し、
確率の最大絶対誤差は`1.49e-8`以下、softmax和の誤差は`1.19e-7`以下でした。

## Transformer block

`just bench-transformer`は決定的なweightを持つgeneric `TransformerBlock`を、同じ
`forward[T : TensorOps]`でCPU/WebNNへ構築します。これは旧feed-forwardベンチとの互換blockです。
固定shapeのgraphは次の演算を合成します。

```text
input [tokens, width]
  → Linear [width, 2×width]
  → LayerNormalization(axis=1)
  → GELU(exact erf definition)
  → Linear [2×width, width]
  → residual add
```

Chrome Canary 152.0.7946.0、headless、WebNN npu、tokens 16、warmup 5回・15 samples:

| width | hidden | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 5 | 0.220 | 0.080 | 0.120 | 2.75x | 1.200 | 0.400 | 55.900 | 0.400 | 1.74e-4 |
| 64 | 128 | 3 | 0.700 | 0.133 | 0.233 | 5.25x | 0.200 | 0.200 | 1.800 | 0.300 | 3.99e-4 |
| 128 | 256 | 1 | 2.600 | 0.100 | 0.200 | 26.00x | 0.300 | 0.100 | 1.100 | 0.000 | 4.92e-4 |
| 256 | 512 | 1 | 9.500 | 0.200 | 0.300 | 47.50x | 0.300 | 0.500 | 1.200 | 0.100 | 6.11e-4 |

widthを`256,128,64,32`へ反転した別processでは、最初のwidth 256だけcompileが53.8 ms、
後続は0.9–1.1 msでした。したがって約55 msはwidth 32固有ではなく、このCanary processで
最初のLayerNormalization/GELU graphをcompileするcold costです。steady-stateでは全shapeで
WebNNが素朴なMoonBit JS CPU参照より高速でした。CPU/WebNNの最大絶対誤差は全shapeで
`6.11e-4`以下です。LayerNormalizationのreduction順序やbackend精度差を含むため、Transformer
blockのdifferential toleranceは`1e-3`とします。

## Self-attention

`just bench-attention`は、同じgeneric `SelfAttention[T : TensorOps]`をCPU/WebNNへ
materializeし、single-headとmulti-headを同じ実装で測定します。モデルAPIは
`[tokens,width]`と`[batch,tokens,width]`の両方を受けます。maskは`none`、`causal`、
`padding`、`causal-padding`から選べます。固定shapeのgraphは次の演算を合成します。

```text
input [tokens, width]
  → Q/K/V Linear
  → reshape + transpose [heads, tokens, head-size]
  → batched matmul(Q, Kᵀ) × 1/√head-size
  → additive mask → softmax
  → batched matmul(attention, V)
  → transpose + reshape → output Linear
```

Chrome Canary 152.0.7946.0、headless、WebNN npu、tokens 16、heads 4、causal mask、
warmup 5回・15 samples:

| width | heads | mask | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | causal | 3 | 0.400 | 0.133 | 0.167 | 3.00x | 1.400 | 0.500 | 54.100 | 0.300 | 2.38e-7 |
| 64 | 4 | causal | 1 | 1.000 | 0.100 | 0.300 | 10.00x | 0.300 | 0.100 | 1.100 | 0.100 | 9.54e-7 |
| 128 | 4 | causal | 1 | 3.200 | 0.200 | 0.300 | 16.00x | 0.300 | 0.300 | 1.100 | 0.000 | 7.63e-6 |
| 256 | 4 | causal | 1 | 11.400 | 0.200 | 0.300 | 57.00x | 0.200 | 0.600 | 1.400 | 0.100 | 1.91e-5 |

width 64で4種のmaskを同一process内で測ると、WebNN p50はいずれも`0.100 ms`、
最大絶対誤差は`9.54e-7`以下でした。CPU参照は最適化済みnative backendではなく素朴な
MoonBit JS実装なので、speedupは実装間の比較です。全batch共通maskは
`[1,tokens,tokens]`、batch別maskは`[batch,1,tokens,tokens]`のadditive constantとして
保持し、必要なbatch/head dimensionへbroadcastします。

## Transformer encoder

`just bench-encoder`は、rank-3入力を受けるgeneric `TransformerEncoderBlock`を測定します。
`LayerNorm`、`SelfAttention`、`FeedForward`を標準的なpre-norm順で合成します。このベンチは
`TransformerEncoderConfig`と`TransformerEncoderParameters`をCPU/WebNN materializerへ渡し、
batchごとにvalid token数が異なる`[batch,1,tokens,tokens]` causal-padding maskを使います。

```text
input [batch, tokens, width]
  → LayerNormalization(last axis)
  → multi-head SelfAttention → residual add(input)
  → LayerNormalization(last axis)
  → Linear [width, 2×width] → GELU
  → Linear [2×width, width] → residual add
```

Chrome Canary 152.0.7946.0、headless、WebNN npu、batch 2、tokens 16、heads 4、
causal-padding mask、warmup 5回・15 samples:

| width | hidden | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 1 | 1.500 | 0.200 | 0.500 | 7.50x | 1.200 | 0.700 | 53.700 | 0.400 | 2.52e-4 |
| 64 | 128 | 1 | 4.000 | 0.300 | 0.400 | 13.33x | 0.300 | 0.200 | 1.200 | 0.000 | 6.68e-4 |
| 128 | 256 | 1 | 12.900 | 0.400 | 0.400 | 32.25x | 0.300 | 0.400 | 1.300 | 0.100 | 7.82e-4 |
| 256 | 512 | 1 | 46.600 | 0.600 | 0.700 | 77.67x | 0.500 | 1.100 | 1.700 | 0.000 | 2.36e-3 |

この測定でもCPU参照は素朴なMoonBit JS実装です。2回のLayerNormalizationと2つのresidualを
通すため、width 256の最大絶対誤差は`2.36e-3`でした。large encoderのdifferential toleranceは
`3e-3`とし、width 256をCanary E2Eへ含めています。各processの最初のencoder compileだけ
`53.7 ms`で、後続shapeは`1.2–1.7 ms`だったため、steady-stateとは分けて扱います。

## Transformer encoder stack

`just bench-encoder-stack`は、各層が異なるweight、bias、LayerNorm parameterを持つ
`TransformerEncoderStack`全体を単一WebNN graphへcompileします。causal-padding maskは
`[batch,1,tokens,tokens]`のconstantを一度だけ作り、全層で共有します。

Chrome Canary 152.0.7946.0、headless、WebNN npu、batch 2、tokens 16、width 64、heads 4、
causal-padding mask、warmup 5回・15 samples:

| layers | hidden | ops/sample | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | context ms | graph ms | compile ms | prepare ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1 | 4.100 | 0.300 | 0.400 | 13.67x | 1.200 | 0.800 | 52.600 | 0.200 | 6.84e-4 |
| 2 | 128 | 1 | 7.900 | 0.500 | 0.600 | 15.80x | 0.200 | 0.400 | 1.500 | 0.000 | 2.23e-3 |
| 4 | 128 | 1 | 15.900 | 0.900 | 1.100 | 17.67x | 0.400 | 0.500 | 2.100 | 0.100 | 7.35e-3 |

CPU/WebNNとも実行時間はほぼ層数に比例しました。4層でも素朴なCPU参照に対して17.67倍ですが、
float32の演算順による差も層ごとに累積し、最大絶対誤差は`7.35e-3`へ増えています。1層目の
compile `52.6 ms`はprocess初回のcold outlierで、2/4層の`1.5/2.1 ms`と分けて扱います。

## BERT post-norm encoder

`just bench-bert-encoder`はHugging Face形式のnamed tensorを検証・転置してから、inference用
`BertEncoderStack`をCPU/WebNNへmaterializeします。通常のBERTと同じpost-norm順、padding mask、
LayerNorm epsilon `1e-12`、exact GELUを使い、loader時間をgraph構築やcompileから分離します。
deterministic fixtureは各層16 tensorを持ち、全層で異なる値を使います。

Chrome Canary 152.0.7946.0、headless、WebNN npu、batch 2、tokens 16、width 64、heads 4、
padding mask、warmup 5回・15 samples:

| layers | tensors | CPU p50 ms | WebNN p50 ms | WebNN p95 ms | speedup | load ms | context ms | graph ms | compile ms | prepare ms | max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 4.500 | 0.400 | 0.500 | 11.25x | 1.700 | 1.500 | 0.900 | 2.800 | 0.400 | 5.96e-7 |
| 2 | 32 | 7.900 | 0.800 | 1.100 | 9.87x | 0.300 | 0.500 | 0.400 | 2.200 | 0.100 | 1.43e-6 |
| 4 | 64 | 15.800 | 1.400 | 1.700 | 11.29x | 0.600 | 0.300 | 0.600 | 2.700 | 0.100 | 4.29e-6 |

WebNN p50は層数にほぼ比例し、4層でも素朴なCPU参照に対して11.29倍でした。post-normで各層の
出力を再正規化するfixtureでは、4層の最大絶対誤差も`4.29e-6`に収まりました。loaderのcold値は
1層目だけ`1.7 ms`で、後続は`0.3–0.6 ms`のため、ファイルdecodeを追加した後は別途反復測定します。

SafeTensors adapter追加後は、8,296-byte・32 tensorの2層fixtureをCanary E2Eで実ファイルとして
fetchし、header/data offset検証、F32 decode、Hugging Face名のmappingと転置を経由して同じ
CPU/WebNN graphを実行しています。上表の`load ms`はnamed tensorからparameterへの変換だけで、
fetchとSafeTensors decodeは含まないため、実モデルfile loaderの性能値とは区別します。
