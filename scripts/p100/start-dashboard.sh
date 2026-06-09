#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Startup script for P100 Testing Dashboard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}P100 Testing Dashboard${NC}"
echo "========================"

# Get script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
FRONTEND_DIR="$REPO_ROOT/frontend"

# Check if running from correct directory
if [ ! -f "$FRONTEND_DIR/app.py" ]; then
    echo -e "${RED}Error: Frontend app not found at $FRONTEND_DIR/app.py${NC}"
    echo "Make sure you're running this script from the repository root."
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check/create virtual environment
VENV_DIR="$REPO_ROOT/.venv_frontend"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -q -r "$FRONTEND_DIR/requirements.txt"

# Check environment variables
echo ""
echo -e "${GREEN}Environment Configuration:${NC}"
echo "-------------------------"

# Load .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    echo -e "${BLUE}Loading environment from .env file...${NC}"
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Check required variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. You may need it to download model weights.${NC}"
fi

if [ -z "$JWT_SECRET" ]; then
    echo -e "${YELLOW}Warning: JWT_SECRET not set. API authentication may fail.${NC}"
fi

# Show configuration
SERVICE_PORT="${SERVICE_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8050}"

echo "Service Port (vLLM): $SERVICE_PORT"
echo "Dashboard Port: $DASHBOARD_PORT"
echo ""

# Start dashboard
echo -e "${GREEN}Starting P100 Testing Dashboard...${NC}"
echo -e "${BLUE}Open your browser at: http://localhost:$DASHBOARD_PORT${NC}"
echo ""

cd "$FRONTEND_DIR"
python3 app.py
