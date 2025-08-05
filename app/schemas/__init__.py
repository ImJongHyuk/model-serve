"""Pydantic schemas package for OpenAI-compatible API models."""

from .chat_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Message,
    ContentPart,
    ImageUrl,
    Choice,
    StreamChoice,
    ChoiceDelta,
    Usage,
)
from .model_models import (
    ModelInfo,
    ModelsResponse,
)
from .error_models import (
    ErrorDetail,
    ErrorResponse,
    OpenAICompatibleException,
    InvalidRequestError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RequestTimeoutError,
    RateLimitError,
    InternalServerError,
    ModelError,
    ServiceUnavailableError,
    ModelNotLoadedError,
    ImageProcessingError,
    ValidationError,
    create_error_response,
    create_validation_error_response,
    create_server_error_response,
)

__all__ = [
    # Chat completion models
    "ChatCompletionRequest",
    "ChatCompletionResponse", 
    "ChatCompletionStreamResponse",
    "Message",
    "ContentPart",
    "ImageUrl",
    "Choice",
    "StreamChoice",
    "ChoiceDelta",
    "Usage",
    # Model endpoint models
    "ModelInfo",
    "ModelsResponse",
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    # Exception classes
    "OpenAICompatibleException",
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "RequestTimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ModelError",
    "ServiceUnavailableError",
    "ModelNotLoadedError",
    "ImageProcessingError",
    "ValidationError",
    # Utility functions
    "create_error_response",
    "create_validation_error_response",
    "create_server_error_response",
]