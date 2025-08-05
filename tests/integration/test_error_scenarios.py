"""Integration tests for error handling scenarios."""

import asyncio
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.api.main import create_application
from app.schemas.error_models import (
    ValidationError,
    ModelError,
    ServiceUnavailableError,
    RequestTimeoutError,
    RateLimitError
)


class TestErrorScenarios:
    """Integration tests for error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('app.config.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings.model_name = "test-model"
            mock_settings.device = "cpu"
            mock_settings.debug = True
            mock_settings.cors_enabled = True
            mock_settings.cors_origins = ["*"]
            mock_settings.cors_allow_credentials = True
            mock_settings.cors_allow_methods = ["*"]
            mock_settings.cors_allow_headers = ["*"]
            mock_settings.trusted_hosts = None
            mock_settings.auth_enabled = False
            mock_settings.api_key = None
            mock_settings_class.return_value = mock_settings
            
            self.app = create_application()
            self.client = TestClient(self.app)
    
    def test_validation_error_handling(self):
        """Test that validation errors are properly handled."""
        # Send invalid request data
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": "invalid_messages_format",  # Should be array
                "temperature": 2.5  # Out of valid range
            }
        )
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert "Invalid request" in data["error"]["message"]
        assert "X-Request-ID" in response.headers
    
    def test_model_error_handling(self):
        """Test that model errors are properly handled."""
        # Mock chat service to raise ModelError
        from app.api.routes.chat import get_chat_service
        from app.services.chat_service import ChatService
        
        mock_chat_service = Mock(spec=ChatService)
        mock_chat_service.create_completion.side_effect = ModelError("Model inference failed")
        
        async def mock_get_chat_service():
            return mock_chat_service
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "model_error"
            assert "Model inference failed" in data["error"]["message"]
            assert "X-Request-ID" in response.headers
        finally:
            self.app.dependency_overrides.clear()
    
    def test_service_unavailable_error_handling(self):
        """Test that service unavailable errors are properly handled."""
        # Mock chat service to raise ServiceUnavailableError
        from app.api.routes.chat import get_chat_service
        
        async def mock_get_chat_service():
            raise ServiceUnavailableError("Service is not available")
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = self.client.get("/v1/models")
            
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "service_unavailable"
            assert "Service is not available" in data["error"]["message"]
            assert "X-Request-ID" in response.headers
        finally:
            self.app.dependency_overrides.clear()
    
    def test_timeout_error_handling(self):
        """Test that timeout errors are properly handled."""
        # Mock chat service to raise TimeoutError
        from app.api.routes.chat import get_chat_service
        
        async def mock_get_chat_service():
            raise asyncio.TimeoutError("Operation timed out")
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = self.client.get("/v1/models")
            
            assert response.status_code == 408
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "timeout_error"
            assert "timed out" in data["error"]["message"].lower()
            assert "X-Request-ID" in response.headers
        finally:
            self.app.dependency_overrides.clear()
    
    def test_http_exception_handling(self):
        """Test that HTTP exceptions are properly handled."""
        # Request non-existent endpoint
        response = self.client.get("/v1/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "http_error"
        assert "X-Request-ID" in response.headers
    
    def test_general_exception_handling(self):
        """Test that unexpected exceptions are properly handled."""
        # Mock chat service to raise unexpected exception
        from app.api.routes.chat import get_chat_service
        
        async def mock_get_chat_service():
            raise ValueError("Unexpected error")
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = self.client.get("/v1/models")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "server_error"
            assert "unexpected error occurred" in data["error"]["message"].lower()
            assert "X-Request-ID" in response.headers
        finally:
            self.app.dependency_overrides.clear()
    
    def test_error_response_format_consistency(self):
        """Test that all error responses follow consistent OpenAI format."""
        # Test validation error format
        response = self.client.post(
            "/v1/chat/completions",
            json={"invalid": "data"}
        )
        
        assert response.status_code >= 400
        data = response.json()
        
        # Check OpenAI-compatible error format
        assert "error" in data
        error = data["error"]
        
        required_fields = ["message", "type"]
        for field in required_fields:
            assert field in error, f"Missing required field: {field}"
        
        # Check that message is a string
        assert isinstance(error["message"], str)
        assert len(error["message"]) > 0
        
        # Check that type follows OpenAI patterns
        valid_types = [
            "invalid_request_error",
            "model_error", 
            "service_unavailable",
            "timeout_error",
            "rate_limit_exceeded",
            "server_error",
            "http_error"
        ]
        assert error["type"] in valid_types
    
    def test_error_logging(self):
        """Test that errors are properly logged."""
        # This test would require checking logs, but we can at least verify
        # that the error handling doesn't break the logging system
        
        # Mock chat service to raise an error
        from app.api.routes.chat import get_chat_service
        
        async def mock_get_chat_service():
            raise ModelError("Test error for logging")
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            # This should log the error without breaking
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            
            # Error should be handled gracefully
            assert response.status_code == 500
            assert "error" in response.json()
        finally:
            self.app.dependency_overrides.clear()
    
    def test_error_headers_consistency(self):
        """Test that error responses include consistent headers."""
        # Test different error types to ensure headers are consistent
        error_endpoints = [
            ("/v1/nonexistent", 404),  # HTTP error
            ("/v1/chat/completions", 422)  # Validation error with invalid data
        ]
        
        for endpoint, expected_status in error_endpoints:
            if expected_status == 422:
                response = self.client.post(endpoint, json={"invalid": "data"})
            else:
                response = self.client.get(endpoint)
            
            assert response.status_code == expected_status
            
            # Check that error responses include required headers
            assert "X-Request-ID" in response.headers
            assert "Content-Type" in response.headers
            assert response.headers["Content-Type"] == "application/json"