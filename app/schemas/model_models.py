"""OpenAI-compatible models endpoint response models."""

from typing import List, Optional
from pydantic import BaseModel, Field
import time


class ModelInfo(BaseModel):
    """Information about a single model."""
    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = Field(default="huggingface", description="Organization that owns the model")
    permission: Optional[List] = Field(default_factory=list, description="Model permissions")
    root: Optional[str] = Field(None, description="Root model if this is a fine-tune")
    parent: Optional[str] = Field(None, description="Parent model if this is a fine-tune")


class ModelsResponse(BaseModel):
    """Response for the models endpoint."""
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models")