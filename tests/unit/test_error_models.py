"""Unit tests for error response models and custom exceptions."""

import pytest
import json
from fastapi import HTTPException
from app.schemas.error_models import (
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


class TestErrorDetail:
    """Test ErrorDetail model."""

    def test_minimal_error_detail(self):
        """Test minimal error detail creation."""
        error = ErrorDetail(message="Test error", type="test_error")
        assert error.message == "Test error"
        assert error.type == "test_error"
        assert error.param is None
        assert error.code is None

    def test_full_error_detail(self):
        """Test error detail with all fields."""
        error = ErrorDetail(
            message="Invalid parameter",
            type="invalid_request_error",
            param="temperature",
            code="invalid_value"
        )
        assert error.message == "Invalid parameter"
        assert error.type == "invalid_request_error"
        assert error.param == "temperature"
        assert error.code == "invalid_value"

    def test_error_detail_serialization(self):
        """Test error detail JSON serialization."""
        error = ErrorDetail(
            message="Test error",
            type="test_error",
            param="test_param"
        )
        json_str = error.model_dump_json()
        data = json.loads(json_str)
        
        assert data["message"] == "Test error"
        assert data["type"] == "test_error"
        assert data["param"] == "test_param"
        assert data["code"] is None


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response_creation(self):
        """Test error response creation."""
        error_detail = ErrorDetail(message="Test error", type="test_error")
        response = ErrorResponse(error=error_detail)
        
        assert response.error == error_detail
        assert response.error.message == "Test error"
        assert response.error.type == "test_error"

    def test_error_response_serialization(self):
        """Test error response JSON serialization."""
        error_detail = ErrorDetail(
            message="Invalid request",
            type="invalid_request_error",
            param="model"
        )
        response = ErrorResponse(error=error_detail)
        
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        
        assert "error" in data
        assert data["error"]["message"] == "Invalid request"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "model"


class TestOpenAICompatibleException:
    """Test base OpenAI compatible exception."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = OpenAICompatibleException(
            status_code=400,
            message="Test error",
            error_type="test_error"
        )
        
        assert exc.status_code == 400
        assert exc.message == "Test error"
        assert exc.error_type == "test_error"
        assert exc.param is None
        assert exc.code is None
        
        # Check that detail is properly formatted
        assert "error" in exc.detail
        assert exc.detail["error"]["message"] == "Test error"
        assert exc.detail["error"]["type"] == "test_error"

    def test_exception_with_all_fields(self):
        """Test exception with all fields."""
        exc = OpenAICompatibleException(
            status_code=400,
            message="Invalid parameter",
            error_type="invalid_request_error",
            param="temperature",
            code="invalid_value",
            headers={"X-Custom": "value"}
        )
        
        assert exc.status_code == 400
        assert exc.param == "temperature"
        assert exc.code == "invalid_value"
        assert exc.headers == {"X-Custom": "value"}
        
        assert exc.detail["error"]["param"] == "temperature"
        assert exc.detail["error"]["code"] == "invalid_value"


class TestSpecificExceptions:
    """Test specific exception classes."""

    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        exc = InvalidRequestError(
            message="Invalid temperature value",
            param="temperature",
            code="invalid_value"
        )
        
        assert exc.status_code == 400
        assert exc.message == "Invalid temperature value"
        assert exc.error_type == "invalid_request_error"
        assert exc.param == "temperature"
        assert exc.code == "invalid_value"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError()
        
        assert exc.status_code == 401
        assert exc.message == "Invalid API key provided"
        assert exc.error_type == "invalid_request_error"
        assert exc.headers == {"WWW-Authenticate": "Bearer"}

    def test_authentication_error_custom_message(self):
        """Test AuthenticationError with custom message."""
        exc = AuthenticationError(message="API key missing", code="missing_key")
        
        assert exc.message == "API key missing"
        assert exc.code == "missing_key"

    def test_permission_error(self):
        """Test PermissionError."""
        exc = PermissionError()
        
        assert exc.status_code == 403
        assert exc.error_type == "invalid_request_error"
        assert "permission" in exc.message.lower()

    def test_not_found_error(self):
        """Test NotFoundError."""
        exc = NotFoundError(message="Model not found")
        
        assert exc.status_code == 404
        assert exc.message == "Model not found"
        assert exc.error_type == "invalid_request_error"

    def test_request_timeout_error(self):
        """Test RequestTimeoutError."""
        exc = RequestTimeoutError(message="Request took too long")
        
        assert exc.status_code == 408
        assert exc.message == "Request took too long"
        assert exc.error_type == "timeout_error"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(message="Too many requests", retry_after=60)
        
        assert exc.status_code == 429
        assert exc.message == "Too many requests"
        assert exc.error_type == "rate_limit_exceeded"
        assert exc.headers == {"Retry-After": "60"}

    def test_rate_limit_error_no_retry_after(self):
        """Test RateLimitError without retry_after."""
        exc = RateLimitError()
        
        assert exc.status_code == 429
        assert exc.headers is None

    def test_internal_server_error(self):
        """Test InternalServerError."""
        exc = InternalServerError(message="Something went wrong")
        
        assert exc.status_code == 500
        assert exc.message == "Something went wrong"
        assert exc.error_type == "server_error"

    def test_model_error(self):
        """Test ModelError."""
        exc = ModelError(message="Model inference failed", code="inference_error")
        
        assert exc.status_code == 500
        assert exc.message == "Model inference failed"
        assert exc.error_type == "server_error"
        assert exc.code == "inference_error"

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        exc = ServiceUnavailableError(
            message="Service down for maintenance",
            retry_after=300
        )
        
        assert exc.status_code == 503
        assert exc.message == "Service down for maintenance"
        assert exc.error_type == "server_error"
        assert exc.headers == {"Retry-After": "300"}

    def test_model_not_loaded_error(self):
        """Test ModelNotLoadedError."""
        exc = ModelNotLoadedError()
        
        assert exc.status_code == 503
        assert exc.message == "Model is not loaded"
        assert exc.error_type == "server_error"

    def test_image_processing_error(self):
        """Test ImageProcessingError."""
        exc = ImageProcessingError(
            message="Invalid image format",
            param="image_url",
            code="invalid_format"
        )
        
        assert exc.status_code == 400
        assert exc.message == "Invalid image format"
        assert exc.error_type == "invalid_request_error"
        assert exc.param == "image_url"
        assert exc.code == "invalid_format"

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError(
            message="Parameter out of range",
            param="temperature",
            code="out_of_range"
        )
        
        assert exc.status_code == 400
        assert exc.message == "Parameter out of range"
        assert exc.error_type == "invalid_request_error"
        assert exc.param == "temperature"
        assert exc.code == "out_of_range"


class TestUtilityFunctions:
    """Test utility functions for creating error responses."""

    def test_create_error_response(self):
        """Test create_error_response function."""
        response = create_error_response(
            status_code=400,
            message="Test error",
            error_type="test_error",
            param="test_param",
            code="test_code"
        )
        
        assert "error" in response
        assert response["error"]["message"] == "Test error"
        assert response["error"]["type"] == "test_error"
        assert response["error"]["param"] == "test_param"
        assert response["error"]["code"] == "test_code"

    def test_create_error_response_minimal(self):
        """Test create_error_response with minimal parameters."""
        response = create_error_response(
            status_code=500,
            message="Server error",
            error_type="server_error"
        )
        
        assert response["error"]["message"] == "Server error"
        assert response["error"]["type"] == "server_error"
        assert response["error"]["param"] is None
        assert response["error"]["code"] is None

    def test_create_validation_error_response(self):
        """Test create_validation_error_response function."""
        response = create_validation_error_response(
            message="Invalid value",
            param="temperature"
        )
        
        assert response["error"]["message"] == "Invalid value"
        assert response["error"]["type"] == "invalid_request_error"
        assert response["error"]["param"] == "temperature"

    def test_create_validation_error_response_no_param(self):
        """Test create_validation_error_response without param."""
        response = create_validation_error_response(message="Invalid request")
        
        assert response["error"]["message"] == "Invalid request"
        assert response["error"]["type"] == "invalid_request_error"
        assert response["error"]["param"] is None

    def test_create_server_error_response(self):
        """Test create_server_error_response function."""
        response = create_server_error_response()
        
        assert response["error"]["message"] == "Internal server error occurred"
        assert response["error"]["type"] == "server_error"

    def test_create_server_error_response_custom_message(self):
        """Test create_server_error_response with custom message."""
        response = create_server_error_response(message="Database connection failed")
        
        assert response["error"]["message"] == "Database connection failed"
        assert response["error"]["type"] == "server_error"


class TestExceptionInheritance:
    """Test exception inheritance and HTTPException compatibility."""

    def test_exceptions_are_http_exceptions(self):
        """Test that all custom exceptions inherit from HTTPException."""
        exceptions = [
            InvalidRequestError("test"),
            AuthenticationError(),
            PermissionError(),
            NotFoundError(),
            RequestTimeoutError(),
            RateLimitError(),
            InternalServerError(),
            ModelError(),
            ServiceUnavailableError(),
            ModelNotLoadedError(),
            ImageProcessingError("test"),
            ValidationError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, HTTPException)
            assert isinstance(exc, OpenAICompatibleException)

    def test_exception_detail_format(self):
        """Test that all exceptions have properly formatted detail."""
        exc = InvalidRequestError("Test message", param="test_param")
        
        assert isinstance(exc.detail, dict)
        assert "error" in exc.detail
        assert "message" in exc.detail["error"]
        assert "type" in exc.detail["error"]
        
        # Should be serializable to JSON
        json_str = json.dumps(exc.detail)
        data = json.loads(json_str)
        assert data["error"]["message"] == "Test message"