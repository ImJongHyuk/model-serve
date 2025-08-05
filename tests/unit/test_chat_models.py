"""Unit tests for chat completion models."""

import pytest
import json
from pydantic import ValidationError
from app.schemas.chat_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ContentPart,
    ImageUrl,
    Choice,
    Usage,
    ChoiceDelta,
    StreamChoice,
    ChatCompletionStreamResponse
)


class TestImageUrl:
    """Test ImageUrl model."""

    def test_valid_image_url(self):
        """Test valid image URL creation."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        assert image_url.url == "https://example.com/image.jpg"
        assert image_url.detail == "auto"

    def test_image_url_with_detail(self):
        """Test image URL with specific detail level."""
        image_url = ImageUrl(url="https://example.com/image.jpg", detail="high")
        assert image_url.detail == "high"

    def test_base64_image_url(self):
        """Test base64 encoded image URL."""
        base64_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        image_url = ImageUrl(url=base64_url)
        assert image_url.url == base64_url


class TestContentPart:
    """Test ContentPart model."""

    def test_text_content_part(self):
        """Test text content part creation."""
        content = ContentPart(type="text", text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"
        assert content.image_url is None

    def test_image_content_part(self):
        """Test image content part creation."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        content = ContentPart(type="image_url", image_url=image_url)
        assert content.type == "image_url"
        assert content.image_url == image_url
        assert content.text is None

    def test_text_content_validation_error(self):
        """Test validation error when text is missing for text type."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type="text")
        assert "text must be provided when type is 'text'" in str(exc_info.value)

    def test_image_content_validation_error(self):
        """Test validation error when image_url is missing for image_url type."""
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type="image_url")
        assert "image_url must be provided when type is 'image_url'" in str(exc_info.value)


class TestMessage:
    """Test Message model."""

    def test_simple_text_message(self):
        """Test simple text message creation."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.name is None

    def test_message_with_name(self):
        """Test message with name field."""
        message = Message(role="user", content="Hello", name="John")
        assert message.name == "John"

    def test_multimodal_message(self):
        """Test message with text and image content."""
        content_parts = [
            ContentPart(type="text", text="What's in this image?"),
            ContentPart(
                type="image_url", 
                image_url=ImageUrl(url="https://example.com/image.jpg")
            )
        ]
        message = Message(role="user", content=content_parts)
        assert message.role == "user"
        assert len(message.content) == 2
        assert message.content[0].type == "text"
        assert message.content[1].type == "image_url"

    def test_empty_text_validation_error(self):
        """Test validation error for empty text content."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user", content="")
        assert "text content cannot be empty" in str(exc_info.value)

    def test_empty_content_list_validation_error(self):
        """Test validation error for empty content list."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user", content=[])
        assert "content list cannot be empty" in str(exc_info.value)

    def test_invalid_role_validation_error(self):
        """Test validation error for invalid role."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Hello")


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        messages = [Message(role="user", content="Hello")]
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages)
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 1
        assert request.temperature == 0.7
        assert request.stream is False

    def test_request_with_all_parameters(self):
        """Test request with all parameters."""
        messages = [Message(role="user", content="Hello")]
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stream=True,
            stop=["\\n", "END"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            logit_bias={"50256": -100},
            user="user123"
        )
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.stop == ["\\n", "END"]
        assert request.presence_penalty == 0.1
        assert request.frequency_penalty == 0.2
        assert request.logit_bias == {"50256": -100}
        assert request.user == "user123"

    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        messages = [Message(role="user", content="Hello")]
        
        # Valid temperature
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, temperature=1.5)
        assert request.temperature == 1.5
        
        # Invalid temperature - too high
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, temperature=3.0)
        
        # Invalid temperature - negative
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, temperature=-0.1)

    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        messages = [Message(role="user", content="Hello")]
        
        # Valid max_tokens
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, max_tokens=100)
        assert request.max_tokens == 100
        
        # Invalid max_tokens - zero or negative
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, max_tokens=0)

    def test_top_p_validation(self):
        """Test top_p parameter validation."""
        messages = [Message(role="user", content="Hello")]
        
        # Valid top_p
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, top_p=0.5)
        assert request.top_p == 0.5
        
        # Invalid top_p - too high
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, top_p=1.1)
        
        # Invalid top_p - zero or negative
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, top_p=0.0)

    def test_stop_validation(self):
        """Test stop parameter validation."""
        messages = [Message(role="user", content="Hello")]
        
        # Valid stop - string
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, stop="END")
        assert request.stop == "END"
        
        # Valid stop - list
        request = ChatCompletionRequest(model="gpt-3.5-turbo", messages=messages, stop=["\\n", "END"])
        assert request.stop == ["\\n", "END"]
        
        # Invalid stop - too many items
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo", 
                messages=messages, 
                stop=["1", "2", "3", "4", "5"]
            )

    def test_empty_messages_validation_error(self):
        """Test validation error for empty messages list."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-3.5-turbo", messages=[])


class TestUsage:
    """Test Usage model."""

    def test_valid_usage(self):
        """Test valid usage creation."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_invalid_total_tokens(self):
        """Test validation error for incorrect total_tokens."""
        with pytest.raises(ValidationError) as exc_info:
            Usage(prompt_tokens=10, completion_tokens=20, total_tokens=25)
        assert "total_tokens must equal prompt_tokens + completion_tokens" in str(exc_info.value)


class TestChoice:
    """Test Choice model."""

    def test_valid_choice(self):
        """Test valid choice creation."""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason="stop")
        assert choice.index == 0
        assert choice.message == message
        assert choice.finish_reason == "stop"

    def test_choice_without_finish_reason(self):
        """Test choice without finish_reason."""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message)
        assert choice.finish_reason is None


class TestChatCompletionResponse:
    """Test ChatCompletionResponse model."""

    def test_valid_response(self):
        """Test valid response creation."""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        response = ChatCompletionResponse(
            model="gpt-3.5-turbo",
            choices=[choice],
            usage=usage
        )
        
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.usage == usage
        assert response.object == "chat.completion"
        assert response.id.startswith("chatcmpl-")
        assert isinstance(response.created, int)

    def test_response_serialization(self):
        """Test response JSON serialization."""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        response = ChatCompletionResponse(
            model="gpt-3.5-turbo",
            choices=[choice],
            usage=usage
        )
        
        # Should be able to serialize to JSON
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        
        assert data["model"] == "gpt-3.5-turbo"
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["usage"]["total_tokens"] == 15


class TestStreamingModels:
    """Test streaming response models."""

    def test_choice_delta(self):
        """Test ChoiceDelta model."""
        delta = ChoiceDelta(role="assistant", content="Hello")
        assert delta.role == "assistant"
        assert delta.content == "Hello"

    def test_stream_choice(self):
        """Test StreamChoice model."""
        delta = ChoiceDelta(content="Hello")
        choice = StreamChoice(index=0, delta=delta, finish_reason=None)
        assert choice.index == 0
        assert choice.delta == delta
        assert choice.finish_reason is None

    def test_stream_response(self):
        """Test ChatCompletionStreamResponse model."""
        delta = ChoiceDelta(content="Hello")
        choice = StreamChoice(index=0, delta=delta)
        
        response = ChatCompletionStreamResponse(
            model="gpt-3.5-turbo",
            choices=[choice]
        )
        
        assert response.model == "gpt-3.5-turbo"
        assert response.object == "chat.completion.chunk"
        assert len(response.choices) == 1
        assert response.id.startswith("chatcmpl-")

    def test_stream_response_serialization(self):
        """Test streaming response JSON serialization."""
        delta = ChoiceDelta(content="Hello")
        choice = StreamChoice(index=0, delta=delta)
        
        response = ChatCompletionStreamResponse(
            model="gpt-3.5-turbo",
            choices=[choice]
        )
        
        # Should be able to serialize to JSON
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        
        assert data["model"] == "gpt-3.5-turbo"
        assert data["object"] == "chat.completion.chunk"
        assert len(data["choices"]) == 1