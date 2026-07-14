set shell := ["bash", "-cu"]

default: check

install:
    pnpm install
    moon update

fixture-mnist source="../nn/data/mnist" count="100":
    MNIST_DATA_DIR="{{ source }}" MNIST_FIXTURE_COUNT="{{ count }}" node scripts/prepare-mnist-fixture.mjs

fixture-tflite:
    node scripts/fetch-tflite-fixtures.mjs

fixture-bert:
    node scripts/prepare-bert-fixture.mjs

fmt:
    moon fmt

info:
    moon info

unit:
    moon test --target js

build:
    moon build --target js --release

e2e: fixture-tflite fixture-bert build
    pnpm exec playwright test

ci-unit:
    pnpm run test:ci-config
    moon fmt --check
    moon check --target js
    moon test --target js
    moon build --target js --release

ci-e2e: fixture-tflite fixture-bert build
    pnpm exec playwright test

bench sizes="32,64,128,256" warmup="10" iterations="30": build
    BENCH_SIZES="{{ sizes }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench.mjs

bench-mnist batches="1,16,100" warmup="10" iterations="30": build
    BENCH_BATCHES="{{ batches }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-mnist.mjs

bench-mnist-cache batches="1,16,100" warmup="10" iterations="30": build
    BENCH_BATCHES="{{ batches }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-mnist-cache.mjs

bench-mnist-pool pools="1,2,4,8" requests="256" warmup="10" iterations="30": build
    BENCH_POOLS="{{ pools }}" BENCH_REQUESTS="{{ requests }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-mnist-pool.mjs

bench-named-io sizes="64,1024,16384" warmup="10" iterations="30": build
    BENCH_SIZES="{{ sizes }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-named-io.mjs

bench-tiny-cnn batches="1,2,4,8,16,32,64" warmup="10" iterations="30" layouts="nchw,nhwc": build
    BENCH_BATCHES="{{ batches }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" BENCH_LAYOUTS="{{ layouts }}" node scripts/bench-tiny-cnn.mjs

bench-transformer widths="32,64,128,256" tokens="16" warmup="10" iterations="30": build
    BENCH_WIDTHS="{{ widths }}" BENCH_TOKENS="{{ tokens }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-transformer.mjs

bench-attention widths="32,64,128,256" tokens="16" heads="4" masks="causal" warmup="10" iterations="30": build
    BENCH_WIDTHS="{{ widths }}" BENCH_TOKENS="{{ tokens }}" BENCH_HEADS="{{ heads }}" BENCH_MASKS="{{ masks }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-attention.mjs

bench-encoder widths="32,64,128,256" batch="2" tokens="16" heads="4" mask="causal-padding" warmup="10" iterations="30": build
    BENCH_WIDTHS="{{ widths }}" BENCH_BATCH_SIZE="{{ batch }}" BENCH_TOKENS="{{ tokens }}" BENCH_HEADS="{{ heads }}" BENCH_MASK="{{ mask }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-encoder.mjs

bench-encoder-stack layers="1,2,4" batch="2" tokens="16" width="64" heads="4" mask="causal-padding" warmup="10" iterations="30": build
    BENCH_LAYERS="{{ layers }}" BENCH_BATCH_SIZE="{{ batch }}" BENCH_TOKENS="{{ tokens }}" BENCH_WIDTH="{{ width }}" BENCH_HEADS="{{ heads }}" BENCH_MASK="{{ mask }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-encoder-stack.mjs

bench-bert-encoder layers="1,2,4" batch="2" tokens="16" width="64" heads="4" mask="padding" warmup="10" iterations="30": fixture-bert build
    BENCH_LAYERS="{{ layers }}" BENCH_BATCH_SIZE="{{ batch }}" BENCH_TOKENS="{{ tokens }}" BENCH_WIDTH="{{ width }}" BENCH_HEADS="{{ heads }}" BENCH_MASK="{{ mask }}" BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-bert-encoder.mjs

bench-litert warmup="10" iterations="30": build
    BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-litert.mjs

bench-tflite warmup="10" iterations="30": build
    BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-tflite.mjs

bench-tflite-runner-cache warmup="10" iterations="30": build
    BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-tflite-runner-cache.mjs

bench-mobilenet-v2 warmup="3" iterations="10": fixture-tflite build
    BENCH_WARMUP="{{ warmup }}" BENCH_ITERATIONS="{{ iterations }}" node scripts/bench-mobilenet-v2.mjs

check: fmt info fixture-tflite fixture-bert
    moon check --target js
    moon test --target js
    moon build --target js --release
    pnpm exec playwright test
