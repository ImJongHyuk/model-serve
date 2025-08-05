"""Configuration management for the FastAPI application."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model configuration
    model_name: str = Field(
        default="skt/A.X-4.0-VL-Light",
        description="Hugging Face model name to serve"
    )
    
    # API configuration
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication"
    )
    auth_enabled: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to"
    )
    port: int = Field(
        default=8000,
        description="Port to bind the server to"
    )
    
    # Performance configuration
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent requests"
    )
    request_timeout: int = Field(
        default=300,
        description="Request timeout in seconds"
    )
    gpu_memory_fraction: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use"
    )
    
    # Generation defaults
    default_temperature: float = Field(
        default=0.7,
        description="Default temperature for text generation"
    )
    default_max_tokens: int = Field(
        default=2048,
        description="Default maximum tokens for generation"
    )
    default_top_p: float = Field(
        default=1.0,
        description="Default top_p for nucleus sampling"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Device configuration
    device: str = Field(
        default="auto",
        description="Device to use for inference (auto, cpu, cuda)"
    )
    
    # FastAPI configuration
    debug: bool = Field(
        default=False,
        description="Enable debug mode with docs endpoints"
    )
    
    # CORS configuration
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Allow credentials in CORS requests"
    )
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "OPTIONS"],
        description="Allowed HTTP methods for CORS"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed headers for CORS"
    )
    
    # Security configuration
    trusted_hosts: Optional[list[str]] = Field(
        default=None,
        description="List of trusted hosts (None to disable)"
    )
    
    # Rate limiting configuration
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per IP"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()