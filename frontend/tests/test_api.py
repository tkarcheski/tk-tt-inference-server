#!/usr/bin/env python3
"""Unit tests for API endpoints without actual server."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMockedAPI:
    """Tests with mocked server manager."""
    
    @patch('api_server.server_manager')
    def test_start_server_mock(self, mock_manager):
        """Test start endpoint with mocked manager."""
        from api_server import start_server
        from pydantic import BaseModel
        
        mock_manager.is_running.return_value = False
        mock_manager.start.return_value = True
        mock_manager.model = "Llama-3.1-8B"
        
        class MockConfig(BaseModel):
            model: str = "Llama-3.1-8B"
            context_length: str = "2048"
        
        config = MockConfig()
        result = start_server(config, None)
        assert result["status"] == "starting"
    
    @patch('api_server.server_manager')
    def test_stop_server_mock(self, mock_manager):
        """Test stop endpoint with mocked manager."""
        from api_server import stop_server
        
        mock_manager.stop.return_value = True
        result = stop_server()
        assert result["status"] == "stopped"
    
    @patch('api_server.server_manager')
    def test_status_running_mock(self, mock_manager):
        """Test status endpoint when running."""
        from api_server import get_status
        
        mock_manager.is_running.return_value = True
        mock_manager.model = "Llama-3.1-8B"
        mock_manager.device = "p100"
        
        result = get_status()
        assert result["running"] is True
        assert result["model"] == "Llama-3.1-8B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
