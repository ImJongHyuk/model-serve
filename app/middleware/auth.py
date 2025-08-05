"""Authentication middleware for API key validation."""

import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import Settings
from app.schemas.error_models import create_error_response


logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication using Authorization header."""
    
    def __init__(self, app, settings: Settings):
        """Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            settings: Application settings containing auth configuration
        """
        super().__init__(app)
        self.settings = settings
        self.enabled = settings.auth_enabled and bool(settings.api_key)
        
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication validation.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip auth for certain endpoints
        if self._should_skip_auth(request):
            return await call_next(request)
        
        # Skip if authentication is disabled
        if not self.enabled:
            return await call_next(request)
        
        # Validate API key
        auth_result = self._validate_api_key(request)
        if not auth_result["valid"]:
            return self._create_auth_error_response(request, auth_result["error"])
        
        # Continue to next middleware/handler
        return await call_next(request)
    
    def _should_skip_auth(self, request: Request) -> bool:
        """Check if authentication should be skipped for this request.
        
        Args:
            request: HTTP request
            
        Returns:
            True if auth should be skipped
        """
        # Skip auth for health checks and docs
        skip_paths = [
            "/health",
            "/",
            "/docs",
            "/redoc", 
            "/openapi.json"
        ]
        
        return request.url.path in skip_paths
    
    def _validate_api_key(self, request: Request) -> dict:
        """Validate API key from Authorization header.
        
        Args:
            request: HTTP request
            
        Returns:
            Dictionary with validation result
        """
        # Get Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return {
                "valid": False,
                "error": {
                    "message": "Authorization header is required",
                    "type": "invalid_request_error",
                    "param": "authorization",
                    "code": "missing_authorization_header"
                }
            }
        
        # Parse Bearer token
        api_key = self._extract_bearer_token(auth_header)
        if not api_key:
            return {
                "valid": False,
                "error": {
                    "message": "Invalid authorization header format. Expected 'Bearer <api_key>'",
                    "type": "invalid_request_error", 
                    "param": "authorization",
                    "code": "invalid_authorization_header"
                }
            }
        
        # Validate API key
        if api_key != self.settings.api_key:
            return {
                "valid": False,
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "param": "authorization", 
                    "code": "invalid_api_key"
                }
            }
        
        return {"valid": True}
    
    def _extract_bearer_token(self, auth_header: str) -> Optional[str]:
        """Extract API key from Bearer token format.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            API key or None if invalid format
        """
        if not auth_header.startswith("Bearer "):
            return None
        
        return auth_header[7:]  # Remove "Bearer " prefix
    
    def _create_auth_error_response(self, request: Request, error: dict) -> JSONResponse:
        """Create authentication error response.
        
        Args:
            request: HTTP request
            error: Error details
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Request {request_id}: Authentication failed - {error['message']}")
        
        error_response = create_error_response(
            status_code=401,
            message=error["message"],
            error_type=error["type"],
            param=error.get("param"),
            code=error.get("code")
        )
        
        return JSONResponse(
            status_code=401,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )