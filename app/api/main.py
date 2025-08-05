"""Main FastAPI application with middleware and configuration."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings
from app.schemas.error_models import (
    OpenAICompatibleException,
    ValidationError,
    InternalServerError,
    ModelError,
    ServiceUnavailableError,
    RequestTimeoutError,
    RateLimitError,
    create_error_response
)
from app.utils.app_state import initialize_app_state, get_app_state, clear_app_state
from app.middleware.auth import AuthenticationMiddleware
from app.api.routes import chat_router, models_router


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Request {request_id}: {response.status_code} "
                f"({process_time:.3f}s)"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id}: Error after {process_time:.3f}s - {str(e)}"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling uncaught exceptions."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        try:
            return await call_next(request)
        except OpenAICompatibleException:
            # Re-raise OpenAI compatible exceptions
            raise
        except Exception as e:
            # Convert unexpected errors to OpenAI format
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise InternalServerError(f"Internal server error: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Hugging Face OpenAI-compatible API server...")
    
    try:
        # Initialize application state
        initialize_app_state()
        
        logger.info("Server startup completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down server...")
        
        # Cleanup services (will be added in later tasks)
        clear_app_state()
        
        logger.info("Server shutdown completed")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        settings: Application settings (optional)
        
    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = Settings()
    
    # Create FastAPI app with OpenAI-compatible metadata
    app = FastAPI(
        title="Hugging Face OpenAI-Compatible API",
        description="OpenAI-compatible API server for Hugging Face A.X-4.0-VL-Light model",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters - last added is executed first)
    
    # Error handling middleware (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Authentication middleware (before business logic)
    app.add_middleware(AuthenticationMiddleware, settings=settings)
    
    # CORS middleware
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
            expose_headers=["X-Request-ID", "X-Process-Time"]
        )
    
    # Trusted host middleware (for production)
    if settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts
        )
    
    # Store settings in app state
    app.state.settings = settings
    
    return app


def setup_exception_handlers(app: FastAPI):
    """Set up global exception handlers.
    
    Args:
        app: FastAPI application
    """
    
    @app.exception_handler(OpenAICompatibleException)
    async def openai_exception_handler(request: Request, exc: OpenAICompatibleException):
        """Handle OpenAI-compatible exceptions.
        
        Args:
            request: HTTP request
            exc: OpenAI-compatible exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors.
        
        Args:
            request: HTTP request
            exc: Validation error
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: Validation error - {exc}")
        
        # Convert Pydantic errors to OpenAI format
        error_details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            error_details.append(f"{field}: {error['msg']}")
        
        error_message = "Invalid request: " + "; ".join(error_details)
        validation_error = ValidationError(error_message)
        
        return JSONResponse(
            status_code=validation_error.status_code,
            content=validation_error.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions.
        
        Args:
            request: HTTP request
            exc: HTTP exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: HTTP {exc.status_code} - {exc.detail}")
        
        # Convert to OpenAI format
        error_response = create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type="http_error",
            code=str(exc.status_code)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions.
        
        Args:
            request: HTTP request
            exc: Exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Request {request_id}: Unexpected error - {exc}", exc_info=True)
        
        # Convert to OpenAI format
        internal_error = InternalServerError("An unexpected error occurred")
        
        return JSONResponse(
            status_code=internal_error.status_code,
            content=internal_error.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
        """Handle asyncio timeout errors.
        
        Args:
            request: HTTP request
            exc: Timeout exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: Request timeout - {exc}")
        
        # Convert to OpenAI format
        timeout_error = RequestTimeoutError("Request processing timed out")
        
        return JSONResponse(
            status_code=timeout_error.status_code,
            content=timeout_error.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(ModelError)
    async def model_exception_handler(request: Request, exc: ModelError):
        """Handle model inference errors.
        
        Args:
            request: HTTP request
            exc: Model error exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Request {request_id}: Model error - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable_exception_handler(request: Request, exc: ServiceUnavailableError):
        """Handle service unavailable errors.
        
        Args:
            request: HTTP request
            exc: Service unavailable exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Request {request_id}: Service unavailable - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
        """Handle rate limiting errors.
        
        Args:
            request: HTTP request
            exc: Rate limit exception
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: Rate limit exceeded - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers={"X-Request-ID": request_id}
        )


def create_application() -> FastAPI:
    """Create the main FastAPI application with all configurations.
    
    Returns:
        Configured FastAPI application
    """
    # Load settings
    settings = Settings()
    
    # Create app
    app = create_app(settings)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include API routers
    app.include_router(chat_router)
    app.include_router(models_router)
    
    # Add basic health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint.
        
        Returns:
            Health status
        """
        app_state = get_app_state()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - app_state.get("startup_time", time.time()),
            "version": "1.0.0"
        }
    
    # Add root endpoint with API information
    @app.get("/")
    async def root():
        """Root endpoint with API information.
        
        Returns:
            API information
        """
        return {
            "name": "Hugging Face OpenAI-Compatible API",
            "version": "1.0.0",
            "description": "OpenAI-compatible API server for Hugging Face A.X-4.0-VL-Light model",
            "endpoints": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health"
            },
            "documentation": "/docs" if settings.debug else None
        }
    
    return app


# Create the application instance
app = create_application()


# Legacy functions for backward compatibility
def get_request_count() -> int:
    """Get the current request count.
    
    Returns:
        Number of requests processed
    """
    from app.utils.app_state import get_request_count as _get_request_count
    return _get_request_count()