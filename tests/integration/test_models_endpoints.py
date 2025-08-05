"""Integration tests for models endpoints."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.api.main import create_application
from app.services.chat_service import ChatService


class TestModelsEndpoints:
    """Integration tests for /v1/models endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock settings
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
            mock_settings_class.return_value = mock_settings
            
            self.app = create_application()
            self.client = TestClient(self.app)
    
    def create_mock_chat_service(self, model_info=None):
        """Create a mock chat service for testing."""
        if model_info is None:
            model_info = {
                "name": "test-model",
                "device": "cpu",
                "created": 1234567890,
                "type": "multimodal",
                "capabilities": {
                    "text_generation": True,
                    "image_understanding": True,
                    "streaming": True
                }
            }
        
        mock_chat_service = Mock(spec=ChatService)
        mock_chat_service.get_model_info = AsyncMock(return_value=model_info)
        return mock_chat_service
    
    def override_dependency(self, mock_chat_service):
        """Override the chat service dependency."""
        from app.api.routes.chat import get_chat_service
        
        async def mock_get_chat_service():
            return mock_chat_service
        
        self.app.dependency_overrides[get_chat_service] = mock_get_chat_service
    
    def teardown_method(self):
        """Clean up after each test."""
        self.app.dependency_overrides.clear()
    
    def test_list_models_success(self):
        """Test successful models listing."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        
        model = data["data"][0]
        assert model["id"] == "test-model"
        assert model["object"] == "model"
        assert model["owned_by"] == "huggingface"
        assert model["created"] == 1234567890
    
    def test_get_model_success(self):
        """Test successful model retrieval."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models/test-model")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"
        assert data["object"] == "model"
        assert data["owned_by"] == "huggingface"
        assert data["created"] == 1234567890
    
    def test_get_model_not_found(self):
        """Test model retrieval with non-existent model."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models/non-existent-model")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert "not found" in data["error"]["message"]
    
    def test_list_models_service_error(self):
        """Test models listing with service error."""
        # Create mock chat service that raises an error
        mock_chat_service = Mock(spec=ChatService)
        from app.schemas.error_models import ServiceUnavailableError
        mock_chat_service.get_model_info = AsyncMock(
            side_effect=ServiceUnavailableError("Service error")
        )
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "service_unavailable"
    
    def test_get_model_service_error(self):
        """Test model retrieval with service error."""
        # Create mock chat service that raises an error
        mock_chat_service = Mock(spec=ChatService)
        from app.schemas.error_models import ServiceUnavailableError
        mock_chat_service.get_model_info = AsyncMock(
            side_effect=ServiceUnavailableError("Service error")
        )
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models/test-model")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "service_unavailable"
    
    def test_service_initialization_failure(self):
        """Test endpoints when service initialization fails."""
        # Don't override dependency, let initialization fail naturally
        response = self.client.get("/v1/models")
        
        # Should get a service unavailable error
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        # The error type might be server_error due to initialization failure
        assert data["error"]["type"] in ["service_unavailable", "server_error"]
        assert "initialization failed" in data["error"]["message"].lower() or "service" in data["error"]["message"].lower()
    
    def test_models_response_headers(self):
        """Test that proper headers are set in models response."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
    
    def test_model_response_headers(self):
        """Test that proper headers are set in single model response."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models/test-model")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
    
    def test_models_endpoint_without_service(self):
        """Test models endpoint behavior when service is not available."""
        # Don't mock the service creation, let it fail naturally
        response = self.client.get("/v1/models")
        
        # Should get a service unavailable error
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
    
    def test_models_openai_compatibility(self):
        """Test that models response is OpenAI-compatible."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check OpenAI-compatible structure
        assert "object" in data
        assert "data" in data
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        
        if data["data"]:
            model = data["data"][0]
            required_fields = ["id", "object", "created", "owned_by"]
            for field in required_fields:
                assert field in model, f"Missing required field: {field}"
            assert model["object"] == "model"
    
    def test_single_model_openai_compatibility(self):
        """Test that single model response is OpenAI-compatible."""
        # Create mock chat service
        mock_chat_service = self.create_mock_chat_service()
        self.override_dependency(mock_chat_service)
        
        response = self.client.get("/v1/models/test-model")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check OpenAI-compatible structure
        required_fields = ["id", "object", "created", "owned_by"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        assert data["object"] == "model"
        assert data["id"] == "test-model"