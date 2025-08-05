"""Unit tests for authentication middleware."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.middleware.auth import AuthenticationMiddleware
from app.config import Settings


class TestAuthenticationMiddleware:
    """Unit tests for AuthenticationMiddleware."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @self.app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}
    
    def test_auth_disabled(self):
        """Test middleware when authentication is disabled."""
        settings = Settings(auth_enabled=False, api_key=None)
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_auth_enabled_no_api_key_configured(self):
        """Test middleware when auth is enabled but no API key is configured."""
        settings = Settings(auth_enabled=True, api_key=None)
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get("/test")
        
        # Should pass through since no API key is configured
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_auth_enabled_with_valid_api_key(self):
        """Test middleware with valid API key."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get(
            "/test",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_auth_enabled_with_invalid_api_key(self):
        """Test middleware with invalid API key."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get(
            "/test",
            headers={"Authorization": "Bearer wrong-api-key"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "invalid_api_key"
    
    def test_auth_enabled_missing_authorization_header(self):
        """Test middleware when Authorization header is missing."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get("/test")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "missing_authorization_header"
    
    def test_auth_enabled_invalid_authorization_format(self):
        """Test middleware with invalid Authorization header format."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        
        # Test various invalid formats
        invalid_headers = [
            "test-api-key",  # Missing Bearer prefix
            "Basic test-api-key",  # Wrong auth type
            "Bearer",  # Missing API key
            "Bearer ",  # Empty API key
        ]
        
        for auth_header in invalid_headers:
            response = client.get(
                "/test",
                headers={"Authorization": auth_header}
            )
            
            assert response.status_code == 401
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "invalid_request_error"
            assert data["error"]["code"] == "invalid_authorization_header"
    
    def test_skip_auth_for_health_endpoint(self):
        """Test that authentication is skipped for health endpoints."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        self.app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        client = TestClient(self.app)
        response = client.get("/health")
        
        # Should succeed without authentication
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_skip_auth_for_docs_endpoints(self):
        """Test that authentication is skipped for documentation endpoints."""
        settings = Settings(auth_enabled=True, api_key="test-api-key")
        
        # Create app with docs enabled
        app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
        app.add_middleware(AuthenticationMiddleware, settings=settings)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        client = TestClient(app)
        
        # Test documentation endpoints
        skip_paths = ["/", "/docs", "/redoc", "/openapi.json"]
        
        for path in skip_paths:
            response = client.get(path)
            # Should not return 401 (may return 404 or redirect, but not auth error)
            assert response.status_code != 401
    
    def test_extract_bearer_token(self):
        """Test bearer token extraction logic."""
        settings = Settings()
        middleware = AuthenticationMiddleware(None, settings)
        
        # Valid cases
        assert middleware._extract_bearer_token("Bearer test-key") == "test-key"
        assert middleware._extract_bearer_token("Bearer abc123") == "abc123"
        
        # Invalid cases
        assert middleware._extract_bearer_token("Basic test-key") is None
        assert middleware._extract_bearer_token("test-key") is None
        assert middleware._extract_bearer_token("Bearer") is None
        assert middleware._extract_bearer_token("") is None
    
    def test_should_skip_auth(self):
        """Test auth skipping logic."""
        settings = Settings()
        middleware = AuthenticationMiddleware(None, settings)
        
        # Create mock requests
        def create_mock_request(path: str):
            request = Mock()
            request.url.path = path
            return request
        
        # Paths that should skip auth
        skip_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
        for path in skip_paths:
            request = create_mock_request(path)
            assert middleware._should_skip_auth(request) is True
        
        # Paths that should require auth
        auth_paths = ["/v1/chat/completions", "/v1/models", "/api/test"]
        for path in auth_paths:
            request = create_mock_request(path)
            assert middleware._should_skip_auth(request) is False