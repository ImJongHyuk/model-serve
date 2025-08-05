"""OpenAI-compatible chat completion request and response models."""

from typing import List, Optional, Union, Literal, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
import time
import uuid


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""
    url: str = Field(..., description="URL of the image or base64 encoded image")
    detail: Optional[Literal["low", "high", "auto"]] = Field(
        default="auto", 
        description="Detail level for image processing"
    )


class ContentPart(BaseModel):
    """Content part that can be text or image."""
    type: Literal["text", "image_url"] = Field(..., description="Type of content")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[ImageUrl] = Field(None, description="Image URL content")

    @model_validator(mode='after')
    def validate_content_part(self):
        """Ensure appropriate content is provided based on type."""
        if self.type == 'text' and not self.text:
            raise ValueError("text must be provided when type is 'text'")
        if self.type == 'image_url' and not self.image_url:
            raise ValueError("image_url must be provided when type is 'image_url'")
        return self


class Message(BaseModel):
    """Chat message with role and content."""
    role: Literal["system", "user", "assistant"] = Field(
        ..., 
        description="Role of the message sender"
    )
    content: Union[str, List[ContentPart]] = Field(
        ..., 
        description="Message content as string or list of content parts"
    )
    name: Optional[str] = Field(None, description="Optional name of the sender")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate content format."""
        if isinstance(v, list):
            if not v:
                raise ValueError("content list cannot be empty")
            # Ensure at least one content part
            for part in v:
                if not isinstance(part, ContentPart):
                    raise ValueError("All content parts must be ContentPart instances")
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("text content cannot be empty")
        return v


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model to use for completion")
    messages: List[Message] = Field(
        ..., 
        min_length=1, 
        description="List of messages in the conversation"
    )
    
    # Generation parameters
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # Response format
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences for generation"
    )
    
    # Additional parameters
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty between -2.0 and 2.0"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty between -2.0 and 2.0"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Logit bias for specific tokens"
    )
    user: Optional[str] = Field(
        default=None,
        description="Unique identifier for the user"
    )

    @field_validator('stop')
    @classmethod
    def validate_stop(cls, v):
        """Validate stop sequences."""
        if isinstance(v, list):
            if len(v) > 4:
                raise ValueError("stop list cannot have more than 4 items")
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("All stop items must be strings")
        return v


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")

    @model_validator(mode='after')
    def validate_total_tokens(self):
        """Ensure total_tokens equals prompt_tokens + completion_tokens."""
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class Choice(BaseModel):
    """A single completion choice."""
    index: int = Field(..., description="Index of the choice")
    message: Message = Field(..., description="The generated message")
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = Field(
        None, 
        description="Reason why the generation stopped"
    )


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:29]}")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")
    system_fingerprint: Optional[str] = Field(
        None, 
        description="System fingerprint for reproducibility"
    )


# Streaming response models
class ChoiceDelta(BaseModel):
    """Delta for streaming responses."""
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming choice."""
    index: int = Field(..., description="Index of the choice")
    delta: ChoiceDelta = Field(..., description="Delta content")
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:29]}")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="Model used for completion")
    choices: List[StreamChoice] = Field(..., description="List of streaming choices")
    system_fingerprint: Optional[str] = Field(
        None, 
        description="System fingerprint for reproducibility"
    )