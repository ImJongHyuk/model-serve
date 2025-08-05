"""OpenAI-compatible error response models and custom exceptions."""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from fastapi import HTTPException
import time


class ErrorDetail(BaseModel):
    """Error detail following OpenAI format."""
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")
    param: Optional[str] = Field(None, description="Parameter that caused the error")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: ErrorDetail = Field(..., description="Error details")


# Custom Exception Classes
class OpenAICompatibleException(HTTPException):
    """Base exception for OpenAI-compatible errors."""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        param: Optional[str] = None,
        code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        
        # Create OpenAI-compatible error response
        error_detail = ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code
        )
        error_response = ErrorResponse(error=error_detail)
        
        super().__init__(
            status_code=status_code,
            detail=error_response.model_dump(),
            headers=headers
        )


class InvalidRequestError(OpenAICompatibleException):
    """400 Bad Request - Invalid parameters or malformed request."""
    
    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=400,
            message=message,
            error_type="invalid_request_error",
            param=param,
            code=code
        )


class AuthenticationError(OpenAICompatibleException):
    """401 Unauthorized - Invalid or missing API key."""
    
    def __init__(
        self,
        message: str = "Invalid API key provided",
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=401,
            message=message,
            error_type="invalid_request_error",
            code=code,
            headers={"WWW-Authenticate": "Bearer"}
        )


class PermissionError(OpenAICompatibleException):
    """403 Forbidden - Insufficient permissions."""
    
    def __init__(
        self,
        message: str = "You don't have permission to access this resource",
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=403,
            message=message,
            error_type="invalid_request_error",
            code=code
        )


class NotFoundError(OpenAICompatibleException):
    """404 Not Found - Resource not found."""
    
    def __init__(
        self,
        message: str = "The requested resource was not found",
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=404,
            message=message,
            error_type="invalid_request_error",
            code=code
        )


class RequestTimeoutError(OpenAICompatibleException):
    """408 Request Timeout - Request processing timeout."""
    
    def __init__(
        self,
        message: str = "Request timed out",
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=408,
            message=message,
            error_type="timeout_error",
            code=code
        )


class RateLimitError(OpenAICompatibleException):
    """429 Too Many Requests - Rate limiting exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        code: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
            
        super().__init__(
            status_code=429,
            message=message,
            error_type="rate_limit_exceeded",
            code=code,
            headers=headers if headers else None
        )


class InternalServerError(OpenAICompatibleException):
    """500 Internal Server Error - Model inference failures."""
    
    def __init__(
        self,
        message: str = "Internal server error occurred",
        code: Optional[str] = None
    ):
        super().__init__(
            status_code=500,
            message=message,
            error_type="server_error",
            code=code
        )


class ModelError(InternalServerError):
    """500 Internal Server Error - Model-specific errors."""
    
    def __init__(
        self,
        message: str = "Model inference failed",
        code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=code
        )


class ServiceUnavailableError(OpenAICompatibleException):
    """503 Service Unavailable - Model not loaded or unavailable."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        code: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
            
        super().__init__(
            status_code=503,
            message=message,
            error_type="server_error",
            code=code,
            headers=headers if headers else None
        )


class ModelNotLoadedError(ServiceUnavailableError):
    """503 Service Unavailable - Model not loaded."""
    
    def __init__(
        self,
        message: str = "Model is not loaded",
        code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=code
        )


class ImageProcessingError(InvalidRequestError):
    """400 Bad Request - Image processing failed."""
    
    def __init__(
        self,
        message: str = "Failed to process image",
        param: str = "image_url",
        code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            param=param,
            code=code
        )


class ValidationError(InvalidRequestError):
    """400 Bad Request - Parameter validation failed."""
    
    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
        code: Optional[str] = None
    ):
        super().__init__(
            message=message,
            param=param,
            code=code
        )


# Utility functions for creating error responses
def create_error_response(
    status_code: int,
    message: str,
    error_type: str,
    param: Optional[str] = None,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized error response dictionary."""
    error_detail = ErrorDetail(
        message=message,
        type=error_type,
        param=param,
        code=code
    )
    error_response = ErrorResponse(error=error_detail)
    return error_response.model_dump()


def create_validation_error_response(
    message: str,
    param: Optional[str] = None
) -> Dict[str, Any]:
    """Create a validation error response."""
    return create_error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        param=param
    )


def create_server_error_response(
    message: str = "Internal server error occurred"
) -> Dict[str, Any]:
    """Create a server error response."""
    return create_error_response(
        status_code=500,
        message=message,
        error_type="server_error"
    )