name = "mizchi/webnn"

version = "0.1.0"

import {
  "mizchi/js@0.12.1",
}

readme = "README.md"

repository = "https://github.com/mizchi/webnn-mbt"

license = "MIT"

keywords = [ "webnn", "moonbit" ]

description = "WebNN backend and TFLite inference runtime for MoonBit"

source = "src"

preferred_target = "js"

options(
  exclude: [
    "examples",
    "fixtures",
    "public",
    "scripts",
    "tests",
    "justfile",
    "moon.work",
    "package.json",
    "playwright.config.ts",
    "pnpm-lock.yaml",
    "src/**/*_wbtest.mbt",
  ],
)
