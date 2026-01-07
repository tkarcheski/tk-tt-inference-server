#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Run benchmarks for GPT-OSS-20b on P100 (Blackhole)
# This is EXPERIMENTAL support for P100 - use at your own risk

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running GPT-OSS-20b benchmarks on P100 (EXPERIMENTAL)${NC}"
echo "=================================================="

# Check if running from repo root
if [ ! -f "run.py" ]; then
    echo -e "${RED}Error: Must run from tt-inference-server repository root${NC}"
    echo "Usage: ./scripts/p100/gpt-oss-20b-benchmark.sh"
    exit 1
fi

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. You may need it for evaluation datasets.${NC}"
    echo "Set it with: export HF_TOKEN=your_token_here"
fi

# Optional: Device ID (defaults to 0)
DEVICE_ID=${DEVICE_ID:-0}
echo "Using device ID: $DEVICE_ID"

# Run benchmarks
echo -e "${GREEN}Running benchmarks...${NC}"
echo "This will start the inference server, run benchmarks, and shut down."
echo ""

python3 run.py \
    --model gpt-oss-20b \
    --device p100 \
    --workflow benchmarks \
    --docker-server \
    --device-id $DEVICE_ID \
    --dev-mode

echo ""
echo -e "${GREEN}Benchmarks completed!${NC}"
echo "Check workflow_logs/benchmarks_output/ for results"
