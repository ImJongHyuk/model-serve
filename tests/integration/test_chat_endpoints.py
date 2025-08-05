"""Integration tests for chat completion endpoints."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.api.main import create_application
from app.services.chat_service import ChatService
from app.schemas.chat_models import (
    ChatCompletionResponse,
    Choice,
    Message,
    Usage
)


class TestChatCompletionEndpoint:
    """Integration tests for /v1/chat/completions endpoint."""
    
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
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_success(self, mock_create_service):
        """Test successful chat completion."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello! How can I help you?"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        )
        mock_chat_service.create_completion = AsyncMock(return_value=mock_response)
        mock_create_service.return_value = mock_chat_service
        
        # Make request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert data["usage"]["total_tokens"] == 18
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_streaming(self, mock_create_service):
        """Test streaming chat completion."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        
        async def mock_stream():
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\\n\\n'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\\n\\n'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\\n\\n'
            yield "data: [DONE]\\n\\n"
        
        mock_chat_service.create_completion_stream = mock_stream
        mock_create_service.return_value = mock_chat_service
        
        # Make streaming request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": True
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "cache-control" in response.headers
        assert response.headers["cache-control"] == "no-cache"
        
        # Check streaming content
        content = response.content.decode()
        assert "Hello" in content
        assert "world" in content
        assert "[DONE]" in content
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_validation_error(self, mock_create_service):
        """Test chat completion with validation error."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        from app.schemas.error_models import ValidationError
        mock_chat_service.create_completion = AsyncMock(
            side_effect=ValidationError("Invalid request")
        )
        mock_create_service.return_value = mock_chat_service
        
        # Make request with invalid data
        request_data = {
            "model": "test-model",
            "messages": []  # Empty messages should cause validation error
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_model_error(self, mock_create_service):
        """Test chat completion with model error."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        from app.schemas.error_models import ModelError
        mock_chat_service.create_completion = AsyncMock(
            side_effect=ModelError("Model generation failed")
        )
        mock_create_service.return_value = mock_chat_service
        
        # Make request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_service_unavailable(self, mock_create_service):
        """Test chat completion with service unavailable."""
        # Mock service creation failure
        from app.schemas.error_models import ServiceUnavailableError
        mock_create_service.side_effect = ServiceUnavailableError("Service not available")
        
        # Make request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "service_unavailable"
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_completion_multimodal(self, mock_create_service):
        """Test chat completion with multimodal input."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="I can see the image."),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25)
        )
        mock_chat_service.create_completion = AsyncMock(return_value=mock_response)
        mock_create_service.return_value = mock_chat_service
        
        # Make request with image
        request_data = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                            }
                        }
                    ]
                }
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "I can see the image."
        assert data["usage"]["total_tokens"] == 25
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_chat_health_check(self, mock_create_service):
        """Test chat health check endpoint."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        mock_chat_service.health_check = AsyncMock(return_value={
            "status": "healthy",
            "model_loaded": True,
            "model_name": "test-model",
            "device": "cpu",
            "memory_usage": {"used": "1GB", "total": "8GB"}
        })
        mock_create_service.return_value = mock_chat_service
        
        response = self.client.get("/v1/chat/completions/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "test-model"
        assert data["device"] == "cpu"
        assert "memory_usage" in data
        assert "request_count" in data
    
    def test_invalid_json_request(self):
        """Test request with invalid JSON."""
        response = self.client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self):
        """Test request with missing required fields."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
            # Missing "model" field
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @patch('app.services.chat_service.create_and_initialize_chat_service')
    def test_request_headers(self, mock_create_service):
        """Test that proper headers are set in response."""
        # Mock chat service
        mock_chat_service = Mock(spec=ChatService)
        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        )
        mock_chat_service.create_completion = AsyncMock(return_value=mock_response)
        mock_create_service.return_value = mock_chat_service
        
        # Make request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers