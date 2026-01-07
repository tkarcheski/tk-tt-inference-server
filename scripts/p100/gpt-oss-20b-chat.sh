#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Quick chat test for GPT-OSS-20b on P100 (Blackhole)
# This is EXPERIMENTAL support for P100 - use at your own risk

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}GPT-OSS-20b Chat Test on P100 (EXPERIMENTAL)${NC}"
echo "=================================================="

# Default settings
SERVICE_PORT=${SERVICE_PORT:-8000}
API_KEY=${JWT_SECRET:-"your-jwt-secret"}

# Simple chat test
echo -e "${BLUE}Sending test prompt to GPT-OSS-20b...${NC}"
echo ""

curl -s -X POST http://localhost:${SERVICE_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What can you tell me about Tenstorrent hardware?"}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }' | python3 -m json.tool 2>/dev/null || echo "Error: Could not parse response. Is the server running?"

echo ""
echo -e "${GREEN}Chat test complete!${NC}"
