"""Chat service for handling chat completion requests and business logic."""

import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

from app.services.model_manager import ModelManager
from app.services.request_validator import RequestValidator
from app.models.response_generator import ResponseGenerator
from app.models.message_processor import MessageProcessor
from app.models.image_handler import ImageHandler
from app.schemas.chat_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    StreamChoice,
    ChoiceDelta,
    Message,
    Usage
)
from app.schemas.error_models import (
    ValidationError,
    ModelError,
    ServiceUnavailableError,
    RequestTimeoutError
)


logger = logging.getLogger(__name__)


class ChatService:
    """Service layer for chat completion operations."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        response_generator: Optional[ResponseGenerator] = None,
        message_processor: Optional[MessageProcessor] = None,
        image_handler: Optional[ImageHandler] = None,
        request_validator: Optional[RequestValidator] = None
    ):
        """Initialize ChatService.
        
        Args:
            model_manager: ModelManager instance
            response_generator: ResponseGenerator instance (optional)
            message_processor: MessageProcessor instance (optional)
            image_handler: ImageHandler instance (optional)
            request_validator: RequestValidator instance (optional)
        """
        self.model_manager = model_manager
        self.message_processor = message_processor or MessageProcessor(model_manager)
        self.image_handler = image_handler or ImageHandler()
        
        # Initialize request validator
        self.request_validator = request_validator or RequestValidator(
            available_model=self.model_manager.model_name
        )
        
        # Initialize response generator with dependencies
        self.response_generator = response_generator or ResponseGenerator(
            model_manager=self.model_manager,
            message_processor=self.message_processor,
            image_handler=self.image_handler
        )
        
        # Service state
        self._initialized = False
        self._request_count = 0
        self._total_tokens_processed = 0
    
    async def initialize(self):
        """Initialize the chat service."""
        if self._initialized:
            return
        
        try:
            # Ensure model is loaded
            if not self.model_manager.is_loaded():
                logger.info("Loading model for ChatService...")
                await self.model_manager.load_model()
            
            # Initialize response generator
            await self.response_generator.initialize()
            
            self._initialized = True
            logger.info("ChatService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatService: {e}")
            raise ServiceUnavailableError(f"Service initialization failed: {str(e)}")
    
    def _ensure_initialized(self):
        """Ensure service is initialized."""
        if not self._initialized:
            raise ServiceUnavailableError("Service not initialized")
    
    async def create_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create a chat completion response.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response
            
        Raises:
            ValidationError: If request is invalid
            ModelError: If model generation fails
            ServiceUnavailableError: If service is not available
        """
        self._ensure_initialized()
        
        try:
            # Validate and normalize request
            request = self._validate_request(request)
            
            # Generate response
            start_time = time.time()
            response_data = await self.response_generator.generate_response(
                request, stream=False
            )
            
            # Create completion response
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())
            
            # Create assistant message
            assistant_message = Message(
                role="assistant",
                content=response_data["text"]
            )
            
            # Create choice
            choice = Choice(
                index=0,
                message=assistant_message,
                finish_reason=response_data.get("finish_reason", "stop")
            )
            
            # Create response
            response = ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=created_timestamp,
                model=request.model,
                choices=[choice],
                usage=response_data["usage"]
            )
            
            # Update service stats
            self._request_count += 1
            self._total_tokens_processed += response_data["usage"].total_tokens
            
            processing_time = time.time() - start_time
            logger.info(
                f"Completed chat completion in {processing_time:.2f}s "
                f"(tokens: {response_data['usage'].total_tokens})"
            )
            
            return response
            
        except ValidationError:
            raise
        except ModelError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_completion: {e}")
            raise ModelError(f"Failed to create completion: {str(e)}")
    
    async def create_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion response.
        
        Args:
            request: Chat completion request
            
        Yields:
            SSE formatted response chunks
            
        Raises:
            ValidationError: If request is invalid
            ModelError: If model generation fails
            ServiceUnavailableError: If service is not available
        """
        self._ensure_initialized()
        
        try:
            # Validate and normalize request
            request = self._validate_request(request)
            
            # Set streaming flag
            request.stream = True
            
            # Generate streaming response
            start_time = time.time()
            
            async for sse_chunk in self.response_generator.generate_sse_stream(request):
                yield sse_chunk
            
            # Update service stats
            self._request_count += 1
            processing_time = time.time() - start_time
            
            logger.info(f"Completed streaming completion in {processing_time:.2f}s")
            
        except ValidationError:
            raise
        except ModelError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_completion_stream: {e}")
            # Send error in SSE format
            from app.models.response_generator import SSEFormatter
            yield SSEFormatter.format_error(f"Failed to create completion: {str(e)}")
    
    def _validate_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Validate and normalize chat completion request.
        
        Args:
            request: Request to validate
            
        Returns:
            Validated and normalized request
            
        Raises:
            ValidationError: If request is invalid
            ServiceUnavailableError: If service is not available
        """
        # Check if model is available
        if not self.model_manager.is_loaded():
            raise ServiceUnavailableError("Model not loaded")
        
        # Use request validator for comprehensive validation
        try:
            validated_request = self.request_validator.validate_and_normalize(request)
            return validated_request
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "initialized": self._initialized,
            "model_loaded": self.model_manager.is_loaded(),
            "model_name": self.model_manager.model_name,
            "request_count": self._request_count,
            "total_tokens_processed": self._total_tokens_processed,
            "model_status": self.model_manager.get_status(),
            "response_generator_stats": self.response_generator.get_generation_stats() if self._initialized else {}
        }
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about supported parameters and their constraints.
        
        Returns:
            Dictionary with parameter information
        """
        return self.request_validator.get_parameter_info()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        try:
            if not self._initialized:
                raise ServiceUnavailableError("Service not initialized")
            
            return {
                "name": self.model_manager.model_name,
                "device": self.model_manager.device,
                "created": int(time.time()),  # Current timestamp as creation time
                "type": "multimodal",
                "capabilities": {
                    "text_generation": True,
                    "image_understanding": True,
                    "streaming": True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise ServiceUnavailableError(f"Failed to retrieve model information: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        try:
            # Check if service is initialized
            if not self._initialized:
                return {
                    "status": "unhealthy",
                    "reason": "Service not initialized",
                    "model_loaded": False
                }
            
            # Check model health
            model_health = await self.model_manager.check_health()
            if not model_health["healthy"]:
                return {
                    "status": "unhealthy",
                    "reason": f"Model unhealthy: {model_health.get('error', 'Unknown error')}",
                    "model_loaded": False,
                    "model_name": self.model_manager.model_name,
                    "device": self.model_manager.device
                }
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_name": self.model_manager.model_name,
                "device": self.model_manager.device,
                "requests_processed": self._request_count,
                "total_tokens": self._total_tokens_processed,
                "memory_usage": model_health.get("memory_usage", {})
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "reason": f"Health check error: {str(e)}",
                "model_loaded": False
            }
    
    async def cleanup(self):
        """Cleanup service resources."""
        cleanup_errors = []
        
        # Cleanup response generator
        if self.response_generator:
            try:
                await self.response_generator.cleanup()
            except Exception as e:
                cleanup_errors.append(f"ResponseGenerator cleanup failed: {e}")
        
        # Cleanup image handler
        if self.image_handler:
            try:
                await self.image_handler.close()
            except Exception as e:
                cleanup_errors.append(f"ImageHandler cleanup failed: {e}")
        
        if cleanup_errors:
            logger.error(f"Errors during ChatService cleanup: {'; '.join(cleanup_errors)}")
        else:
            logger.info("ChatService cleaned up successfully")


# Utility functions
def create_chat_service(
    model_name: str = "skt/A.X-4.0-VL-Light",
    device: str = "auto"
) -> ChatService:
    """Create a ChatService instance with default configuration.
    
    Args:
        model_name: Name of the model to load
        device: Device to use for model
        
    Returns:
        ChatService instance
    """
    model_manager = ModelManager(model_name=model_name)
    return ChatService(model_manager=model_manager)


async def create_and_initialize_chat_service(
    model_name: str = "skt/A.X-4.0-VL-Light",
    device: str = "auto"
) -> ChatService:
    """Create and initialize a ChatService instance.
    
    Args:
        model_name: Name of the model to load
        device: Device to use for model
        
    Returns:
        Initialized ChatService instance
    """
    service = create_chat_service(model_name, device)
    await service.initialize()
    return service