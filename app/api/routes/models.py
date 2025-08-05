"""Models API endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Request, Depends

from app.schemas.model_models import ModelsResponse, ModelInfo
from app.schemas.error_models import ServiceUnavailableError
from app.services.chat_service import ChatService
from app.api.routes.chat import get_chat_service


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
) -> ModelsResponse:
    """List available models.
    
    Returns information about the available model in OpenAI-compatible format.
    
    Args:
        request: FastAPI request object
        chat_service: Chat service dependency
        
    Returns:
        List of available models
        
    Raises:
        ServiceUnavailableError: If service is not available
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        # Get model information from chat service
        model_info = await chat_service.get_model_info()
        
        # Create model info in OpenAI format
        model = ModelInfo(
            id=model_info["name"],
            object="model",
            created=model_info.get("created", 1234567890),  # Placeholder timestamp
            owned_by="huggingface",
            permission=[],
            root=model_info["name"],
            parent=None
        )
        
        logger.info(f"Request {request_id}: Listed models - {model.id}")
        
        return ModelsResponse(
            object="list",
            data=[model]
        )
        
    except Exception as e:
        logger.error(f"Request {request_id}: Failed to list models - {e}")
        raise ServiceUnavailableError(f"Failed to retrieve model information: {str(e)}")


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service)
) -> ModelInfo:
    """Get information about a specific model.
    
    Args:
        model_id: Model identifier
        request: FastAPI request object
        chat_service: Chat service dependency
        
    Returns:
        Model information
        
    Raises:
        ValidationError: If model not found
        ServiceUnavailableError: If service is not available
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        # Get model information from chat service
        model_info = await chat_service.get_model_info()
        
        # Check if requested model matches available model
        if model_id != model_info["name"]:
            from app.schemas.error_models import ValidationError
            raise ValidationError(
                f"Model '{model_id}' not found. Available model: '{model_info['name']}'"
            )
        
        # Create model info in OpenAI format
        model = ModelInfo(
            id=model_info["name"],
            object="model",
            created=model_info.get("created", 1234567890),  # Placeholder timestamp
            owned_by="huggingface",
            permission=[],
            root=model_info["name"],
            parent=None
        )
        
        logger.info(f"Request {request_id}: Retrieved model info - {model.id}")
        
        return model
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        logger.error(f"Request {request_id}: Failed to get model {model_id} - {e}")
        raise ServiceUnavailableError(f"Failed to retrieve model information: {str(e)}")