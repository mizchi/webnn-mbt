import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const output = resolve("fixtures/bert/tiny-bert-encoder.safetensors");
const layers = 2;
const width = 8;
const intermediateSize = 16;

function matrix(rows, columns, seed) {
  return Array.from(
    { length: rows * columns },
    (_, index) => ((index * 13 + seed * 7) % 23 - 11) / 64,
  );
}

function bias(size, seed) {
  return Array.from(
    { length: size },
    (_, index) => ((index * 3 + seed) % 9 - 4) / 256,
  );
}

function normalizationScale(size, seed) {
  return Array.from(
    { length: size },
    (_, index) => 1 + ((index + seed) % 5 - 2) / 64,
  );
}

const tensors = new Map();
for (let layer = 0; layer < layers; layer += 1) {
  const prefix = `bert.encoder.layer.${layer}`;
  const seed = layer * 17;
  tensors.set(`${prefix}.attention.self.query.weight`, {
    shape: [width, width],
    values: matrix(width, width, seed),
  });
  tensors.set(`${prefix}.attention.self.query.bias`, {
    shape: [width],
    values: bias(width, seed),
  });
  tensors.set(`${prefix}.attention.self.key.weight`, {
    shape: [width, width],
    values: matrix(width, width, seed + 1),
  });
  tensors.set(`${prefix}.attention.self.key.bias`, {
    shape: [width],
    values: bias(width, seed + 1),
  });
  tensors.set(`${prefix}.attention.self.value.weight`, {
    shape: [width, width],
    values: matrix(width, width, seed + 2),
  });
  tensors.set(`${prefix}.attention.self.value.bias`, {
    shape: [width],
    values: bias(width, seed + 2),
  });
  tensors.set(`${prefix}.attention.output.dense.weight`, {
    shape: [width, width],
    values: matrix(width, width, seed + 3),
  });
  tensors.set(`${prefix}.attention.output.dense.bias`, {
    shape: [width],
    values: bias(width, seed + 3),
  });
  tensors.set(`${prefix}.attention.output.LayerNorm.weight`, {
    shape: [width],
    values: normalizationScale(width, seed),
  });
  tensors.set(`${prefix}.attention.output.LayerNorm.bias`, {
    shape: [width],
    values: bias(width, seed + 4),
  });
  tensors.set(`${prefix}.intermediate.dense.weight`, {
    shape: [intermediateSize, width],
    values: matrix(intermediateSize, width, seed + 5),
  });
  tensors.set(`${prefix}.intermediate.dense.bias`, {
    shape: [intermediateSize],
    values: bias(intermediateSize, seed + 5),
  });
  tensors.set(`${prefix}.output.dense.weight`, {
    shape: [width, intermediateSize],
    values: matrix(width, intermediateSize, seed + 6),
  });
  tensors.set(`${prefix}.output.dense.bias`, {
    shape: [width],
    values: bias(width, seed + 6),
  });
  tensors.set(`${prefix}.output.LayerNorm.weight`, {
    shape: [width],
    values: normalizationScale(width, seed + 1),
  });
  tensors.set(`${prefix}.output.LayerNorm.bias`, {
    shape: [width],
    values: bias(width, seed + 7),
  });
}

const header = {
  __metadata__: {
    format: "pt",
    fixture: "webnn-mbt-tiny-bert-encoder",
    layers: String(layers),
    width: String(width),
    heads: "2",
    intermediate_size: String(intermediateSize),
  },
};
let dataOffset = 0;
for (const [name, tensor] of tensors) {
  const byteLength = tensor.values.length * 4;
  header[name] = {
    dtype: "F32",
    shape: tensor.shape,
    data_offsets: [dataOffset, dataOffset + byteLength],
  };
  dataOffset += byteLength;
}

let headerText = JSON.stringify(header);
const padding = (8 - (headerText.length % 8)) % 8;
headerText += " ".repeat(padding);
const headerBytes = new TextEncoder().encode(headerText);
const bytes = Buffer.alloc(8 + headerBytes.length + dataOffset);
bytes.writeBigUInt64LE(BigInt(headerBytes.length), 0);
Buffer.from(headerBytes).copy(bytes, 8);
let offset = 8 + headerBytes.length;
for (const tensor of tensors.values()) {
  for (const value of tensor.values) {
    bytes.writeFloatLE(value, offset);
    offset += 4;
  }
}

await mkdir(dirname(output), { recursive: true });
let changed = true;
try {
  changed = !Buffer.from(await readFile(output)).equals(bytes);
} catch {
  // The deterministic fixture is created below.
}
if (changed) await writeFile(output, bytes);
const sha256 = createHash("sha256").update(bytes).digest("hex");
console.log(
  `${changed ? "Generated" : "Validated"} BERT SafeTensors fixture: ` +
    `${bytes.length} bytes, ${tensors.size} tensors, sha256=${sha256}`,
);
