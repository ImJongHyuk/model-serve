"""Chat completions API endpoints."""

import logging
from typing import Union

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat_models import ChatCompletionRequest, ChatCompletionResponse
from app.schemas.error_models import ValidationError, ModelError, ServiceUnavailableError
from app.services.chat_service import ChatService, create_and_initialize_chat_service
from app.utils.app_state import get_app_state, increment_request_count


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1", tags=["chat"])


async def get_chat_service(request: Request) -> ChatService:
    """Dependency to get or create ChatService instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ChatService instance
        
    Raises:
        ServiceUnavailableError: If chat service cannot be initialized
    """
    app_state = get_app_state()
    
    # Check if chat service already exists
    if "chat_service" not in app_state:
        try:
            # Initialize chat service using utility function
            settings = request.app.state.settings
            chat_service = await create_and_initialize_chat_service(
                model_name=settings.model_name,
                device=settings.device
            )
            app_state["chat_service"] = chat_service
            logger.info("Chat service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise ServiceUnavailableError(f"Chat service initialization failed: {str(e)}")
    
    return app_state["chat_service"]


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Create a chat completion.
    
    This endpoint is compatible with OpenAI's chat completions API.
    Supports both streaming and non-streaming responses.
    
    Args:
        request_data: Chat completion request data
        request: FastAPI request object
        chat_service: Chat service dependency
        
    Returns:
        Chat completion response or streaming response
        
    Raises:
        ValidationError: If request validation fails
        ModelError: If model inference fails
        ServiceUnavailableError: If service is not available
    """
    # Increment request counter
    increment_request_count()
    
    # Get request ID for logging
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        f"Request {request_id}: Chat completion - "
        f"model={request_data.model}, messages={len(request_data.messages)}, "
        f"stream={request_data.stream}, max_tokens={request_data.max_tokens}"
    )
    
    try:
        if request_data.stream:
            # Streaming response
            logger.info(f"Request {request_id}: Starting streaming response")
            
            async def generate_stream():
                """Generate streaming response."""
                try:
                    async for chunk in chat_service.create_completion_stream(request_data):
                        yield chunk
                except Exception as e:
                    logger.error(f"Request {request_id}: Streaming error - {e}")
                    # Send error in SSE format
                    error_chunk = f"data: {{\"error\": {{\"message\": \"{str(e)}\", \"type\": \"server_error\"}}}}\\n\\n"
                    yield error_chunk
                    yield "data: [DONE]\\n\\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id
                }
            )
        else:
            # Non-streaming response
            logger.info(f"Request {request_id}: Starting non-streaming response")
            
            response = await chat_service.create_completion(request_data)
            
            logger.info(
                f"Request {request_id}: Completion successful - "
                f"tokens={response.usage.total_tokens}, "
                f"finish_reason={response.choices[0].finish_reason}"
            )
            
            return response
            
    except ValidationError as e:
        logger.warning(f"Request {request_id}: Validation error - {e}")
        raise e
    except ModelError as e:
        logger.error(f"Request {request_id}: Model error - {e}")
        raise e
    except ServiceUnavailableError as e:
        logger.error(f"Request {request_id}: Service unavailable - {e}")
        raise e
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error - {e}", exc_info=True)
        raise ModelError(f"Unexpected error during completion: {str(e)}")


@router.get("/chat/completions/health")
async def chat_health_check(
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
) -> dict:
    """Health check for chat service.
    
    Args:
        request: FastAPI request object
        chat_service: Chat service dependency
        
    Returns:
        Health status information
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        # Get service health status
        health_info = await chat_service.health_check()
        
        logger.info(f"Request {request_id}: Chat health check - {health_info['status']}")
        
        return {
            "status": health_info["status"],
            "model_loaded": health_info.get("model_loaded", False),
            "model_name": health_info.get("model_name", "unknown"),
            "device": health_info.get("device", "unknown"),
            "memory_usage": health_info.get("memory_usage", {}),
            "request_count": get_app_state().get("request_count", 0)
        }
        
    except Exception as e:
        logger.error(f"Request {request_id}: Health check failed - {e}")
        raise ServiceUnavailableError(f"Health check failed: {str(e)}")