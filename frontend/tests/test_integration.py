#!/usr/bin/env python3
"""Integration tests for P100 Dashboard API."""

import pytest
import httpx
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8051"
VLLM_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_client():
    """Create HTTP client for testing."""
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


@pytest.fixture(scope="module")
def vllm_client():
    """Create HTTP client for vLLM server."""
    return httpx.Client(base_url=VLLM_URL, timeout=120.0)


class TestAPIHealth:
    """Test API health endpoints."""
    
    def test_api_health(self, api_client):
        """Test that API is healthy."""
        response = api_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "api_version" in data


class TestServerLifecycle:
    """Integration tests for server start/stop."""
    
    def test_start_server(self, api_client):
        """Test starting the server."""
        config = {
            "model": "Llama-3.1-8B",
            "context_length": "2048"
        }
        response = api_client.post("/api/server/start", json=config)
        assert response.status_code in [200, 202]
        data = response.json()
        assert data["status"] in ["starting", "already_running"]
        assert data["model"] == "Llama-3.1-8B"
    
    def test_server_status_running(self, api_client):
        """Test getting server status while running."""
        # Wait for server to start
        time.sleep(60)
        
        response = api_client.get("/api/server/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert data["model"] == "Llama-3.1-8B"
    
    def test_server_reachable(self, vllm_client):
        """Test that vLLM server is actually reachable."""
        response = vllm_client.get("/health")
        assert response.status_code == 200


class TestChatAPI:
    """Integration tests for chat/completion API."""
    
    def test_chat_completion(self, api_client):
        """Test sending a chat completion request."""
        request = {
            "model": "Llama-3.1-8B",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        response = api_client.post("/api/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert len(data["text"]) > 0
        assert "tokens" in data
        assert data["tokens"] > 0
    
    def test_chat_completion_longer_prompt(self, api_client):
        """Test chat with longer prompt."""
        request = {
            "model": "Llama-3.1-8B",
            "messages": [{"role": "user", "content": "Write a short poem about artificial intelligence."}],
            "max_tokens": 100,
            "temperature": 0.8
        }
        response = api_client.post("/api/chat", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert len(data["text"]) > 10


class TestHardware:
    """Test hardware monitoring endpoints."""
    
    def test_get_hardware(self, api_client):
        """Test getting hardware info."""
        response = api_client.get("/api/hardware")
        assert response.status_code == 200
        data = response.json()
        assert "tt_smi" in data
        assert "tt_smi_list" in data


class TestLogs:
    """Test log endpoints."""
    
    def test_get_logs(self, api_client):
        """Test getting server logs."""
        response = api_client.get("/api/logs?lines=50")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data


class TestServerStop:
    """Test stopping the server."""
    
    def test_stop_server(self, api_client):
        """Test stopping the server."""
        response = api_client.post("/api/server/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
    
    def test_server_status_stopped(self, api_client):
        """Test that server is stopped."""
        time.sleep(5)
        response = api_client.get("/api/server/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
