import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const source = resolve(process.env.MNIST_DATA_DIR ?? "../nn/data/mnist");
const output = resolve("fixtures/mnist");
const count = Number(process.env.MNIST_FIXTURE_COUNT ?? 100);
const spec = { inputSize: 784, hiddenSize: 128, outputSize: 10 };

if (!Number.isInteger(count) || count <= 0 || count > 10_000) {
  throw new Error("MNIST_FIXTURE_COUNT must be between 1 and 10000");
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function readIdxHeader(bytes, expectedMagic, headerSize) {
  if (bytes.byteLength < headerSize) throw new Error("IDX file is truncated");
  const magic = bytes.readUInt32BE(0);
  if (magic !== expectedMagic) {
    throw new Error(`IDX magic mismatch: expected ${expectedMagic}, got ${magic}`);
  }
  return bytes.readUInt32BE(4);
}

const [weights, sourceImages, sourceLabels] = await Promise.all([
  readFile(resolve(source, "mlp_784_128_10.bin")),
  readFile(resolve(source, "t10k-images-idx3-ubyte")),
  readFile(resolve(source, "t10k-labels-idx1-ubyte")),
]);

const expectedWeightBytes =
  (spec.inputSize * spec.hiddenSize +
    spec.hiddenSize +
    spec.hiddenSize * spec.outputSize +
    spec.outputSize) *
  4;
if (weights.byteLength !== expectedWeightBytes) {
  throw new Error(
    `weight length mismatch: expected ${expectedWeightBytes}, got ${weights.byteLength}`,
  );
}

const imageCount = readIdxHeader(sourceImages, 2051, 16);
const labelCount = readIdxHeader(sourceLabels, 2049, 8);
const rows = sourceImages.readUInt32BE(8);
const columns = sourceImages.readUInt32BE(12);
if (rows * columns !== spec.inputSize) {
  throw new Error(`image shape mismatch: ${rows}x${columns}`);
}
if (count > imageCount || count > labelCount) {
  throw new Error(`requested ${count} samples from ${imageCount}/${labelCount}`);
}

const images = Buffer.from(sourceImages.subarray(0, 16 + count * spec.inputSize));
images.writeUInt32BE(count, 4);
const labels = Buffer.from(sourceLabels.subarray(0, 8 + count));
labels.writeUInt32BE(count, 4);

await mkdir(output, { recursive: true });
await Promise.all([
  writeFile(resolve(output, "mlp_784_128_10.bin"), weights),
  writeFile(resolve(output, "t10k-images-idx3-ubyte"), images),
  writeFile(resolve(output, "t10k-labels-idx1-ubyte"), labels),
]);

const manifest = {
  formatVersion: 1,
  count,
  rows,
  columns,
  ...spec,
  files: {
    weights: {
      name: "mlp_784_128_10.bin",
      bytes: weights.byteLength,
      sha256: sha256(weights),
    },
    images: {
      name: "t10k-images-idx3-ubyte",
      bytes: images.byteLength,
      sha256: sha256(images),
    },
    labels: {
      name: "t10k-labels-idx1-ubyte",
      bytes: labels.byteLength,
      sha256: sha256(labels),
    },
  },
};
await writeFile(
  resolve(output, "manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);

console.log(
  `Prepared ${count} MNIST samples and ${weights.byteLength} weight bytes in ${output}`,
);
