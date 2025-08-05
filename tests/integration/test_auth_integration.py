"""Integration tests for authentication middleware."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.api.main import create_application


class TestAuthenticationIntegration:
    """Integration tests for authentication middleware with API endpoints."""
    
    def create_app_with_auth(self, auth_enabled: bool = True, api_key: str = "test-api-key"):
        """Create application with authentication settings."""
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
            mock_settings.auth_enabled = auth_enabled
            mock_settings.api_key = api_key
            mock_settings_class.return_value = mock_settings
            
            return create_application()
    
    def test_chat_endpoint_without_auth(self):
        """Test chat endpoint when authentication is disabled."""
        app = self.create_app_with_auth(auth_enabled=False)
        client = TestClient(app)
        
        # Mock the chat service to avoid model loading
        from app.api.routes.chat import get_chat_service
        from app.services.chat_service import ChatService
        
        mock_chat_service = Mock(spec=ChatService)
        mock_chat_service.create_completion = Mock(return_value={
            "id": "test-completion",
            "object": "chat.completion", 
            "model": "test-model",
            "choices": [],
            "usage": {"total_tokens": 0}
        })
        
        async def mock_get_chat_service():
            return mock_chat_service
        
        app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            
            # Should succeed without authentication
            assert response.status_code != 401  # Not an auth error
        finally:
            app.dependency_overrides.clear()
    
    def test_models_endpoint_with_auth_enabled_valid_key(self):
        """Test models endpoint with valid API key."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        # Mock the chat service
        from app.api.routes.chat import get_chat_service
        from app.services.chat_service import ChatService
        
        mock_chat_service = Mock(spec=ChatService)
        mock_chat_service.get_model_info = Mock(return_value={
            "name": "test-model",
            "device": "cpu",
            "created": 1234567890
        })
        
        async def mock_get_chat_service():
            return mock_chat_service
        
        app.dependency_overrides[get_chat_service] = mock_get_chat_service
        
        try:
            response = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer secret-key"}
            )
            
            # Should succeed with valid API key
            assert response.status_code != 401  # Not an auth error
        finally:
            app.dependency_overrides.clear()
    
    def test_models_endpoint_with_auth_enabled_invalid_key(self):
        """Test models endpoint with invalid API key."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "invalid_api_key"
    
    def test_models_endpoint_with_auth_enabled_missing_header(self):
        """Test models endpoint with missing Authorization header."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        response = client.get("/v1/models")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "missing_authorization_header"
    
    def test_health_endpoint_always_accessible(self):
        """Test that health endpoint is always accessible regardless of auth settings."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        # Should work without authorization header
        response = client.get("/health")
        assert response.status_code == 200
        
        # Should also work with authorization header
        response = client.get(
            "/health",
            headers={"Authorization": "Bearer secret-key"}
        )
        assert response.status_code == 200
        
        # Should work even with wrong API key
        response = client.get(
            "/health", 
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 200
    
    def test_root_endpoint_always_accessible(self):
        """Test that root endpoint is always accessible."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        response = client.get("/")
        assert response.status_code == 200
    
    def test_auth_error_response_format(self):
        """Test that authentication errors follow OpenAI format."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        response = client.get("/v1/models")
        
        assert response.status_code == 401
        data = response.json()
        
        # Check OpenAI-compatible error format
        assert "error" in data
        error = data["error"]
        assert "message" in error
        assert "type" in error
        assert "param" in error
        assert "code" in error
        
        # Check specific error details
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "authorization"
        assert error["code"] == "missing_authorization_header"
    
    def test_auth_headers_in_response(self):
        """Test that authentication responses include proper headers."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="secret-key")
        client = TestClient(app)
        
        response = client.get("/v1/models")
        
        assert response.status_code == 401
        assert "X-Request-ID" in response.headers
    
    def test_multiple_endpoints_with_same_auth_config(self):
        """Test that authentication works consistently across multiple endpoints."""
        app = self.create_app_with_auth(auth_enabled=True, api_key="test-key")
        client = TestClient(app)
        
        api_endpoints = ["/v1/models", "/v1/models/test-model"]
        
        for endpoint in api_endpoints:
            # Without auth - should fail
            response = client.get(endpoint)
            assert response.status_code == 401
            
            # With valid auth - should not fail with auth error
            response = client.get(
                endpoint,
                headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code != 401  # May fail for other reasons, but not auth