"""Pytest configuration for dashboard tests."""

import pytest
import sys
from pathlib import Path

# Add frontend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running server)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API tests."""
    return "http://localhost:8051"


@pytest.fixture(scope="session")
def vllm_base_url():
    """Base URL for vLLM server tests."""
    return "http://localhost:8000"
