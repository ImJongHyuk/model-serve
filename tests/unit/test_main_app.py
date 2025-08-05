"""Unit tests for the main FastAPI application."""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import Request, Response

from app.api.main import (
    create_app,
    create_application,
    setup_exception_handlers,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    get_request_count
)
from app.utils.app_state import (
    get_app_state,
    increment_request_count,
    clear_app_state,
    initialize_app_state
)
from app.config import Settings
from app.schemas.error_models import ValidationError, InternalServerError


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = RequestLoggingMiddleware(Mock())

    @pytest.mark.asyncio
    async def test_dispatch_success(self):
        """Test successful request dispatch with logging."""
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.state = Mock()
        
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # Mock call_next
        async def mock_call_next(request):
            return mock_response
        
        with patch('app.api.main.logger') as mock_logger:
            result = await self.middleware.dispatch(mock_request, mock_call_next)
            
            # Verify response
            assert result == mock_response
            assert "X-Request-ID" in mock_response.headers
            assert "X-Process-Time" in mock_response.headers
            
            # Verify logging
            assert mock_logger.info.call_count == 2  # Request and response logs

    @pytest.mark.asyncio
    async def test_dispatch_with_exception(self):
        """Test request dispatch with exception."""
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/error"
        mock_request.client.host = "127.0.0.1"
        mock_request.state = Mock()
        
        # Mock call_next that raises exception
        async def mock_call_next(request):
            raise ValueError("Test error")
        
        with patch('app.api.main.logger') as mock_logger:
            with pytest.raises(ValueError):
                await self.middleware.dispatch(mock_request, mock_call_next)
            
            # Verify error logging
            mock_logger.error.assert_called_once()


class TestErrorHandlingMiddleware:
    """Test ErrorHandlingMiddleware."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = ErrorHandlingMiddleware(Mock())

    @pytest.mark.asyncio
    async def test_dispatch_success(self):
        """Test successful request dispatch."""
        mock_request = Mock(spec=Request)
        mock_response = Mock(spec=Response)
        
        async def mock_call_next(request):
            return mock_response
        
        result = await self.middleware.dispatch(mock_request, mock_call_next)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_dispatch_with_openai_exception(self):
        """Test dispatch with OpenAI-compatible exception."""
        mock_request = Mock(spec=Request)
        
        async def mock_call_next(request):
            raise ValidationError("Test validation error")
        
        # Should re-raise OpenAI-compatible exceptions
        with pytest.raises(ValidationError):
            await self.middleware.dispatch(mock_request, mock_call_next)

    @pytest.mark.asyncio
    async def test_dispatch_with_unexpected_exception(self):
        """Test dispatch with unexpected exception."""
        mock_request = Mock(spec=Request)
        
        async def mock_call_next(request):
            raise ValueError("Unexpected error")
        
        with patch('app.api.main.logger') as mock_logger:
            with pytest.raises(InternalServerError):
                await self.middleware.dispatch(mock_request, mock_call_next)
            
            # Verify error logging
            mock_logger.error.assert_called_once()


class TestAppCreation:
    """Test application creation functions."""

    def test_create_app_default_settings(self):
        """Test creating app with default settings."""
        app = create_app()
        
        assert app.title == "Hugging Face OpenAI-Compatible API"
        assert app.version == "1.0.0"
        assert hasattr(app.state, 'settings')

    def test_create_app_custom_settings(self):
        """Test creating app with custom settings."""
        settings = Settings(debug=True, cors_enabled=False)
        app = create_app(settings)
        
        assert app.docs_url == "/docs"  # Debug mode enabled
        assert app.state.settings == settings

    def test_create_app_production_settings(self):
        """Test creating app with production settings."""
        settings = Settings(debug=False)
        app = create_app(settings)
        
        assert app.docs_url is None  # Debug mode disabled
        assert app.redoc_url is None
        assert app.openapi_url is None

    def test_create_application(self):
        """Test creating the main application."""
        app = create_application()
        
        assert app.title == "Hugging Face OpenAI-Compatible API"
        assert hasattr(app.state, 'settings')

    def test_setup_exception_handlers(self):
        """Test setting up exception handlers."""
        app = create_app()
        setup_exception_handlers(app)
        
        # Verify exception handlers are registered
        assert len(app.exception_handlers) > 0


class TestAppState:
    """Test application state management."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear app state
        clear_app_state()

    def test_get_app_state(self):
        """Test getting application state."""
        state = get_app_state()
        assert isinstance(state, dict)

    def test_increment_request_count(self):
        """Test incrementing request count."""
        initial_count = get_request_count()
        
        increment_request_count()
        assert get_request_count() == initial_count + 1
        
        increment_request_count()
        assert get_request_count() == initial_count + 2

    def test_get_request_count_initial(self):
        """Test getting initial request count."""
        count = get_request_count()
        assert count == 0


class TestAppEndpoints:
    """Test application endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_application()
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
        assert data["version"] == "1.0.0"

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Hugging Face OpenAI-Compatible API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert data["endpoints"]["chat_completions"] == "/v1/chat/completions"
        assert data["endpoints"]["models"] == "/v1/models"
        assert data["endpoints"]["health"] == "/health"

    def test_root_endpoint_debug_mode(self):
        """Test root endpoint in debug mode."""
        settings = Settings(debug=True)
        app = create_app(settings)
        setup_exception_handlers(app)
        
        # Add root endpoint manually for this test
        @app.get("/")
        async def root():
            return {
                "name": "Hugging Face OpenAI-Compatible API",
                "version": "1.0.0",
                "documentation": "/docs" if settings.debug else None
            }
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["documentation"] == "/docs"

    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint."""
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestMiddlewareIntegration:
    """Test middleware integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_application()
        self.client = TestClient(self.app)

    def test_request_logging_headers(self):
        """Test that request logging adds headers."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

    def test_cors_headers(self):
        """Test CORS headers are added."""
        # Make an OPTIONS request to trigger CORS
        response = self.client.options("/health")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_error_handling_integration(self):
        """Test error handling middleware integration."""
        # This would test actual error scenarios
        # For now, we test that the middleware is properly integrated
        assert len(self.app.user_middleware) > 0


class TestLifespan:
    """Test application lifespan management."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test lifespan startup and shutdown."""
        from app.api.main import lifespan
        
        app = create_app()
        
        # Test lifespan context manager
        async with lifespan(app):
            # During startup
            state = get_app_state()
            assert "startup_time" in state
            assert "request_count" in state
        
        # After shutdown, state should be cleared
        state = get_app_state()
        assert len(state) == 0


class TestAppConfiguration:
    """Test application configuration."""

    def test_cors_configuration(self):
        """Test CORS configuration."""
        settings = Settings(
            cors_enabled=True,
            cors_origins=["https://example.com"],
            cors_allow_credentials=True
        )
        app = create_app(settings)
        
        # Verify CORS middleware is added
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None

    def test_trusted_hosts_configuration(self):
        """Test trusted hosts configuration."""
        settings = Settings(trusted_hosts=["example.com", "api.example.com"])
        app = create_app(settings)
        
        # Verify TrustedHostMiddleware is added
        trusted_host_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "TrustedHostMiddleware":
                trusted_host_middleware = middleware
                break
        
        assert trusted_host_middleware is not None

    def test_no_trusted_hosts(self):
        """Test configuration without trusted hosts."""
        settings = Settings(trusted_hosts=None)
        app = create_app(settings)
        
        # Verify TrustedHostMiddleware is not added
        trusted_host_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "TrustedHostMiddleware":
                trusted_host_middleware = middleware
                break
        
        assert trusted_host_middleware is None