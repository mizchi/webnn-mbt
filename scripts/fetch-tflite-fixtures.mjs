import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const output = resolve("fixtures/tflite");
const fixtures = [
  {
    name: "micro_speech_quantized.tflite",
    bytes: 18_800,
    sha256: "09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83",
    source:
      "https://raw.githubusercontent.com/tensorflow/tflite-micro/fddd3707a3c5733af4cb866f18650441e6712504/tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite",
  },
  {
    name: "audio_preprocessor_int8.tflite",
    bytes: 8_772,
    sha256: "278949d197166fb8b580c0bdc94e902fb709fec0569dcf5766816b28285440e5",
    source:
      "https://raw.githubusercontent.com/tensorflow/tflite-micro/fddd3707a3c5733af4cb866f18650441e6712504/tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8.tflite",
  },
  {
    name: "person_detect.tflite",
    bytes: 300_568,
    sha256: "808cfdfc0cf3a6fa6f6fa26bfa379ea97c16d5db7334637766e39c3408502e9d",
    source:
      "https://raw.githubusercontent.com/tensorflow/tflite-micro/fddd3707a3c5733af4cb866f18650441e6712504/tensorflow/lite/micro/models/person_detect.tflite",
  },
  {
    name: "mobilenet_v2_1.0_224_inat_bird_quant.tflite",
    bytes: 3_531_296,
    sha256: "1e2dc9ffed295b4f2e3c555769971777ac8fb37bb0ad5e6dab506d8da0b42890",
    source:
      "https://raw.githubusercontent.com/google-coral/edgetpu/5020de9386ff370dcc1f63291a2d0f98eeb98adb/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
  },
];

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function loadOrFetch(fixture) {
  const file = resolve(output, fixture.name);
  try {
    const bytes = await readFile(file);
    if (bytes.byteLength === fixture.bytes && sha256(bytes) === fixture.sha256) {
      return false;
    }
  } catch {
    // Fetch below when the fixture is absent.
  }
  const response = await fetch(fixture.source);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${fixture.name}: ${response.status}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.byteLength !== fixture.bytes || sha256(bytes) !== fixture.sha256) {
    throw new Error(`Integrity check failed for ${fixture.name}`);
  }
  await writeFile(file, bytes);
  return true;
}

await mkdir(output, { recursive: true });
let downloaded = 0;
for (const fixture of fixtures) {
  if (await loadOrFetch(fixture)) downloaded += 1;
}
console.log(
  `Validated ${fixtures.length} TFLite fixture(s); downloaded ${downloaded} into ${output}`,
);
