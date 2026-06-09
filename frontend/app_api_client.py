"""API client for Dash frontend to communicate with FastAPI backend."""

import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8051"


class DashboardAPIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def start_server(self, model: str, context_length: str) -> Dict[str, Any]:
        """Start the vLLM server."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/server/start",
                json={"model": model, "context_length": context_length},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    def stop_server(self) -> Dict[str, Any]:
        """Stop the vLLM server."""
        try:
            response = self.session.post(f"{self.base_url}/api/server/stop", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            raise
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status."""
        try:
            response = self.session.get(f"{self.base_url}/api/server/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"running": False, "error": str(e)}
    
    def chat(self, model: str, messages: list, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """Send chat completion request."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def get_hardware(self) -> Dict[str, str]:
        """Get hardware info."""
        try:
            response = self.session.get(f"{self.base_url}/api/hardware", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get hardware: {e}")
            raise
    
    def get_logs(self, lines: int = 100) -> str:
        """Get server logs."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/logs",
                params={"lines": lines},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("logs", "")
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return f"Error: {e}"
    
    def list_models(self) -> list:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/models", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Global API client instance
api_client = DashboardAPIClient()
