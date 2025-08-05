"""Unit tests for ChatService."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.services.chat_service import ChatService, create_chat_service, create_and_initialize_chat_service
from app.services.model_manager import ModelManager
from app.services.request_validator import RequestValidator
from app.models.response_generator import ResponseGenerator
from app.models.message_processor import MessageProcessor
from app.models.image_handler import ImageHandler
from app.schemas.chat_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Usage
)
from app.schemas.error_models import (
    ValidationError,
    ModelError,
    ServiceUnavailableError
)


class TestChatService:
    """Test ChatService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.model_name = "test-model"
        self.mock_model_manager.is_loaded.return_value = True
        self.mock_model_manager.get_status.return_value = {"status": "loaded"}
        
        self.mock_response_generator = Mock(spec=ResponseGenerator)
        self.mock_message_processor = Mock(spec=MessageProcessor)
        self.mock_image_handler = Mock(spec=ImageHandler)
        self.mock_request_validator = Mock(spec=RequestValidator)
        
        self.service = ChatService(
            model_manager=self.mock_model_manager,
            response_generator=self.mock_response_generator,
            message_processor=self.mock_message_processor,
            image_handler=self.mock_image_handler,
            request_validator=self.mock_request_validator
        )

    def test_init_with_all_dependencies(self):
        """Test ChatService initialization with all dependencies."""
        assert self.service.model_manager == self.mock_model_manager
        assert self.service.response_generator == self.mock_response_generator
        assert self.service.message_processor == self.mock_message_processor
        assert self.service.image_handler == self.mock_image_handler
        assert self.service.request_validator == self.mock_request_validator
        assert not self.service._initialized

    def test_init_with_minimal_dependencies(self):
        """Test ChatService initialization with minimal dependencies."""
        service = ChatService(model_manager=self.mock_model_manager)
        
        assert service.model_manager == self.mock_model_manager
        assert service.response_generator is not None
        assert service.message_processor is not None
        assert service.image_handler is not None
        assert service.request_validator is not None

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful service initialization."""
        self.mock_model_manager.load_model = AsyncMock()
        self.mock_response_generator.initialize = AsyncMock()
        
        await self.service.initialize()
        
        assert self.service._initialized
        self.mock_response_generator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_model_not_loaded(self):
        """Test initialization when model is not loaded."""
        self.mock_model_manager.is_loaded.return_value = False
        self.mock_model_manager.load_model = AsyncMock()
        self.mock_response_generator.initialize = AsyncMock()
        
        await self.service.initialize()
        
        assert self.service._initialized
        self.mock_model_manager.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        self.service._initialized = True
        self.mock_response_generator.initialize = AsyncMock()
        
        await self.service.initialize()
        
        # Should not call initialize again
        self.mock_response_generator.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure."""
        self.mock_model_manager.is_loaded.return_value = False
        self.mock_model_manager.load_model = AsyncMock(side_effect=Exception("Load failed"))
        
        with pytest.raises(ServiceUnavailableError):
            await self.service.initialize()
        
        assert not self.service._initialized

    def test_ensure_initialized_success(self):
        """Test _ensure_initialized when service is initialized."""
        self.service._initialized = True
        
        # Should not raise exception
        self.service._ensure_initialized()

    def test_ensure_initialized_failure(self):
        """Test _ensure_initialized when service is not initialized."""
        self.service._initialized = False
        
        with pytest.raises(ServiceUnavailableError):
            self.service._ensure_initialized()

    @pytest.mark.asyncio
    async def test_create_completion_success(self):
        """Test successful completion creation."""
        # Setup
        self.service._initialized = True
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        # Mock validator to return validated request
        self.mock_request_validator.validate_and_normalize.return_value = request
        
        mock_response_data = {
            "text": "Hi there!",
            "finish_reason": "stop",
            "usage": Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        }
        
        self.mock_response_generator.generate_response = AsyncMock(return_value=mock_response_data)
        
        # Execute
        response = await self.service.create_completion(request)
        
        # Verify
        assert isinstance(response, ChatCompletionResponse)
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hi there!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 8
        
        # Check service stats updated
        assert self.service._request_count == 1
        assert self.service._total_tokens_processed == 8
        
        # Check validator was called
        self.mock_request_validator.validate_and_normalize.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_create_completion_not_initialized(self):
        """Test completion creation when service not initialized."""
        self.service._initialized = False
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(ServiceUnavailableError):
            await self.service.create_completion(request)

    @pytest.mark.asyncio
    async def test_create_completion_validation_error(self):
        """Test completion creation with validation error."""
        self.service._initialized = True
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        # Mock validator to raise ValidationError
        self.mock_request_validator.validate_and_normalize.side_effect = ValidationError("Invalid request")
        
        with pytest.raises(ValidationError):
            await self.service.create_completion(request)

    @pytest.mark.asyncio
    async def test_create_completion_model_error(self):
        """Test completion creation with model error."""
        self.service._initialized = True
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        self.mock_response_generator.generate_response = AsyncMock(
            side_effect=ModelError("Generation failed")
        )
        
        with pytest.raises(ModelError):
            await self.service.create_completion(request)

    @pytest.mark.asyncio
    async def test_create_completion_stream_success(self):
        """Test successful streaming completion creation."""
        self.service._initialized = True
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True
        )
        
        # Mock validator to return validated request
        self.mock_request_validator.validate_and_normalize.return_value = request
        
        # Mock SSE stream
        async def mock_sse_stream(req):
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
            yield "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"
            yield "data: [DONE]\n\n"
        
        self.mock_response_generator.generate_sse_stream = mock_sse_stream
        
        # Execute
        chunks = []
        async for chunk in self.service.create_completion_stream(request):
            chunks.append(chunk)
        
        # Verify
        assert len(chunks) == 3
        assert "Hello" in chunks[0]
        assert "world" in chunks[1]
        assert "[DONE]" in chunks[2]
        
        # Check service stats updated
        assert self.service._request_count == 1
        
        # Check validator was called
        self.mock_request_validator.validate_and_normalize.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_create_completion_stream_not_initialized(self):
        """Test streaming completion when service not initialized."""
        self.service._initialized = False
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True
        )
        
        with pytest.raises(ServiceUnavailableError):
            chunks = []
            async for chunk in self.service.create_completion_stream(request):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_create_completion_stream_error(self):
        """Test streaming completion with error."""
        self.service._initialized = True
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True
        )
        
        async def mock_error_stream(req):
            raise Exception("Stream failed")
        
        self.mock_response_generator.generate_sse_stream = mock_error_stream
        
        # Execute
        chunks = []
        async for chunk in self.service.create_completion_stream(request):
            chunks.append(chunk)
        
        # Should receive error in SSE format
        assert len(chunks) == 1
        assert "error" in chunks[0]

    def test_validate_request_success(self):
        """Test successful request validation."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        # Mock validator to return validated request
        self.mock_request_validator.validate_and_normalize.return_value = request
        
        result = self.service._validate_request(request)
        
        assert result == request
        self.mock_request_validator.validate_and_normalize.assert_called_once_with(request)

    def test_validate_request_model_not_loaded(self):
        """Test validation when model not loaded."""
        self.mock_model_manager.is_loaded.return_value = False
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(ServiceUnavailableError):
            self.service._validate_request(request)

    def test_validate_request_validation_error(self):
        """Test validation with validation error from validator."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        # Mock validator to raise ValidationError
        self.mock_request_validator.validate_and_normalize.side_effect = ValidationError("Invalid request")
        
        with pytest.raises(ValidationError):
            self.service._validate_request(request)

    def test_get_service_stats(self):
        """Test getting service statistics."""
        self.service._initialized = True
        self.service._request_count = 5
        self.service._total_tokens_processed = 100
        
        self.mock_response_generator.get_generation_stats.return_value = {
            "model_loaded": True,
            "requests_processed": 5
        }
        
        stats = self.service.get_service_stats()
        
        assert stats["initialized"] is True
        assert stats["model_loaded"] is True
        assert stats["model_name"] == "test-model"
        assert stats["request_count"] == 5
        assert stats["total_tokens_processed"] == 100
        assert "response_generator_stats" in stats

    def test_get_service_stats_not_initialized(self):
        """Test getting service statistics when not initialized."""
        self.service._initialized = False
        
        stats = self.service.get_service_stats()
        
        assert stats["initialized"] is False
        assert stats["response_generator_stats"] == {}

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        mock_param_info = {
            "constraints": {"temperature": {"min": 0.0, "max": 2.0}},
            "limits": {"max_message_count": 100},
            "available_model": "test-model"
        }
        
        self.mock_request_validator.get_parameter_info.return_value = mock_param_info
        
        result = self.service.get_parameter_info()
        
        assert result == mock_param_info
        self.mock_request_validator.get_parameter_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        self.service._initialized = True
        self.service._request_count = 10
        self.service._total_tokens_processed = 200
        
        self.mock_model_manager.check_health = AsyncMock(return_value={
            "healthy": True
        })
        
        health = await self.service.health_check()
        
        assert health["status"] == "healthy"
        assert health["model_name"] == "test-model"
        assert health["requests_processed"] == 10
        assert health["total_tokens"] == 200

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when service not initialized."""
        self.service._initialized = False
        
        health = await self.service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "not initialized" in health["reason"]

    @pytest.mark.asyncio
    async def test_health_check_model_unhealthy(self):
        """Test health check when model is unhealthy."""
        self.service._initialized = True
        
        self.mock_model_manager.check_health = AsyncMock(return_value={
            "healthy": False,
            "error": "Model error"
        })
        
        health = await self.service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "Model unhealthy" in health["reason"]

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check with exception."""
        self.service._initialized = True
        
        self.mock_model_manager.check_health = AsyncMock(
            side_effect=Exception("Health check failed")
        )
        
        health = await self.service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "Health check error" in health["reason"]

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        self.mock_response_generator.cleanup = AsyncMock()
        self.mock_image_handler.close = AsyncMock()
        
        await self.service.cleanup()
        
        self.mock_response_generator.cleanup.assert_called_once()
        self.mock_image_handler.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self):
        """Test cleanup with error."""
        self.mock_response_generator.cleanup = AsyncMock(
            side_effect=Exception("Cleanup failed")
        )
        self.mock_image_handler.close = AsyncMock()
        
        # Should not raise exception
        await self.service.cleanup()
        
        self.mock_response_generator.cleanup.assert_called_once()
        # image_handler.close should still be called even if response_generator.cleanup fails
        self.mock_image_handler.close.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_chat_service_default(self):
        """Test creating ChatService with default parameters."""
        with patch('app.services.chat_service.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            service = create_chat_service()
            
            assert isinstance(service, ChatService)
            mock_manager_class.assert_called_once_with(
                model_name="skt/A.X-4.0-VL-Light",
                device="auto"
            )

    def test_create_chat_service_custom(self):
        """Test creating ChatService with custom parameters."""
        with patch('app.services.chat_service.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            service = create_chat_service(
                model_name="custom-model",
                device="cuda:0"
            )
            
            assert isinstance(service, ChatService)
            mock_manager_class.assert_called_once_with(
                model_name="custom-model",
                device="cuda:0"
            )

    @pytest.mark.asyncio
    async def test_create_and_initialize_chat_service(self):
        """Test creating and initializing ChatService."""
        with patch('app.services.chat_service.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.model_name = "test-model"
            mock_manager.is_loaded.return_value = True
            mock_manager_class.return_value = mock_manager
            
            with patch.object(ChatService, 'initialize', new_callable=AsyncMock) as mock_init:
                service = await create_and_initialize_chat_service()
                
                assert isinstance(service, ChatService)
                mock_init.assert_called_once()


class TestChatServiceIntegration:
    """Integration tests for ChatService."""

    @pytest.mark.asyncio
    async def test_full_completion_flow(self):
        """Test complete completion flow."""
        # Create mocks
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.model_name = "test-model"
        mock_model_manager.is_loaded.return_value = True
        mock_model_manager.get_status.return_value = {"status": "loaded"}
        mock_model_manager.load_model = AsyncMock()
        mock_model_manager.check_health = AsyncMock(return_value={"healthy": True})
        
        mock_response_generator = Mock(spec=ResponseGenerator)
        mock_response_generator.initialize = AsyncMock()
        mock_response_generator.generate_response = AsyncMock(return_value={
            "text": "Hello! How can I help you?",
            "finish_reason": "stop",
            "usage": Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        })
        mock_response_generator.get_generation_stats.return_value = {
            "model_loaded": True,
            "requests_processed": 1
        }
        
        # Create service
        service = ChatService(
            model_manager=mock_model_manager,
            response_generator=mock_response_generator
        )
        
        # Initialize
        await service.initialize()
        
        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100
        )
        
        # Execute
        response = await service.create_completion(request)
        
        # Verify
        assert response.model == "test-model"
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.usage.total_tokens == 18
        
        # Check health
        health = await service.health_check()
        assert health["status"] == "healthy"
        
        # Check stats
        stats = service.get_service_stats()
        assert stats["request_count"] == 1
        assert stats["total_tokens_processed"] == 18
        
        # Cleanup
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling_flow(self):
        """Test error handling in service flow."""
        # Create mocks
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.model_name = "test-model"
        mock_model_manager.is_loaded.return_value = False  # Model not loaded
        
        service = ChatService(model_manager=mock_model_manager)
        
        # Try to create completion without initialization
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(ServiceUnavailableError):
            await service.create_completion(request)
        
        # Check health should show unhealthy
        health = await service.health_check()
        assert health["status"] == "unhealthy"