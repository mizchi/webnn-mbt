# MNIST fixture

Chrome Canary headless E2E用の固定fixtureです。

- `mlp_784_128_10.bin`: `784 → 128 → 10` MLPのlittle-endian float32 weights
- `t10k-images-idx3-ubyte`: MNIST test setの先頭100画像
- `t10k-labels-idx1-ubyte`: 対応する100 labels
- `manifest.json`: shape、byte length、SHA-256

binaryを直接編集せず、ローカルの `mizchi/nn` datasetから再生成します。

```bash
just fixture-mnist ../nn/data/mnist 100
```

重みの格納順は `weight1 / bias1 / weight2 / bias2`、行列shapeはそれぞれ `[784, 128]` と `[128, 10]` です。画像はMoonBit loaderで `pixel / 255` に正規化します。
