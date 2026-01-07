#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Start GPT-OSS-20b inference server on P100 (Blackhole)
# This is EXPERIMENTAL support for P100 - use at your own risk

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting GPT-OSS-20b inference server on P100 (EXPERIMENTAL)${NC}"
echo "=================================================="

# Check if running from repo root
if [ ! -f "run.py" ]; then
    echo -e "${RED}Error: Must run from tt-inference-server repository root${NC}"
    echo "Usage: ./scripts/p100/gpt-oss-20b-server.sh"
    exit 1
fi

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. You may need it to download model weights.${NC}"
    echo "Set it with: export HF_TOKEN=your_token_here"
fi

# Optional: Device ID (defaults to 0)
DEVICE_ID=${DEVICE_ID:-0}
echo "Using device ID: $DEVICE_ID"

# Run the inference server
echo -e "${GREEN}Starting server...${NC}"
echo "This may take several minutes for first-time setup (downloading weights, compiling kernels)"
echo ""

python3 run.py \
    --model gpt-oss-20b \
    --device p100 \
    --workflow server \
    --docker-server \
    --device-id $DEVICE_ID \
    --dev-mode

echo ""
echo -e "${GREEN}Server stopped.${NC}"
