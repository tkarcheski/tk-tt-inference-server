#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Fast pre-flight for train_lora.py: a few optimizer steps on CPU with a tiny
# Qwen base. Run this before committing to a long training run — it exercises
# the full pipeline (dataset load, chat templating, LoRA wrap, save) in a few
# minutes instead of hours.
set -euo pipefail
cd "$(dirname "$0")"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export MAX_STEPS="${MAX_STEPS:-5}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
export OUTPUT_DIR="${OUTPUT_DIR:-out/smoke-test}"

echo "Smoke test: $BASE_MODEL, $MAX_STEPS steps -> $OUTPUT_DIR"
python3 train_lora.py

test -f "$OUTPUT_DIR/adapter_config.json" \
    && echo "OK: adapter written to $OUTPUT_DIR" \
    || { echo "FAIL: no adapter found in $OUTPUT_DIR"; exit 1; }
