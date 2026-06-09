#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Test runner for P100 Dashboard API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}P100 Dashboard API Test Suite${NC}"
echo "=============================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
FRONTEND_DIR="$REPO_ROOT/frontend"

# Check if running from correct directory
if [ ! -f "$FRONTEND_DIR/app.py" ]; then
    echo -e "${RED}Error: Frontend app not found at $FRONTEND_DIR/app.py${NC}"
    exit 1
fi

# Activate virtual environment
VENV_DIR="$REPO_ROOT/.venv_frontend"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -q -r "$FRONTEND_DIR/requirements.txt"

# Check if API server is running
echo ""
echo -e "${YELLOW}Checking if API server is running...${NC}"
if curl -s http://localhost:8051/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server is running on port 8051${NC}"
else
    echo -e "${YELLOW}⚠ API server not running. Starting it...${NC}"
    cd "$FRONTEND_DIR"
    python3 api_server.py &
    API_PID=$!
    
    # Wait for server to start
    echo "Waiting for API server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8051/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API server started${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:8051/api/health > /dev/null 2>&1; then
        echo -e "${RED}✗ Failed to start API server${NC}"
        exit 1
    fi
fi

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
echo "----------------"
cd "$FRONTEND_DIR"

# Run integration tests
pytest tests/test_integration.py -v --tb=short

# Run coverage report (optional)
if [ "$1" == "--coverage" ]; then
    echo ""
    echo -e "${BLUE}Running coverage report...${NC}"
    pytest tests/ --cov=frontend --cov-report=html --cov-report=term
fi

# Cleanup
if [ -n "$API_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping API server...${NC}"
    kill $API_PID 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Test suite complete!${NC}"
