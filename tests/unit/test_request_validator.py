"""Unit tests for RequestValidator and ParameterValidator."""

import pytest
from unittest.mock import Mock

from app.services.request_validator import (
    ParameterValidator,
    RequestValidator,
    create_request_validator,
    validate_chat_request
)
from app.schemas.chat_models import ChatCompletionRequest, Message, ContentPart, ImageUrl
from app.schemas.error_models import ValidationError


class TestParameterValidator:
    """Test ParameterValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

    def test_init(self):
        """Test ParameterValidator initialization."""
        assert self.validator.validation_errors == []

    def test_validate_request_success(self):
        """Test successful request validation."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        result = self.validator.validate_request(request)
        
        assert result.temperature == 0.7
        assert result.max_tokens == 100
        assert result.top_p == 0.9
        assert result.presence_penalty == 0.0  # Default
        assert result.frequency_penalty == 0.0  # Default

    def test_validate_request_with_defaults(self):
        """Test request validation with default values."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        result = self.validator.validate_request(request)
        
        assert result.temperature == ParameterValidator.DEFAULT_TEMPERATURE
        assert result.max_tokens == ParameterValidator.DEFAULT_MAX_TOKENS
        assert result.top_p == ParameterValidator.DEFAULT_TOP_P
        assert result.presence_penalty == ParameterValidator.DEFAULT_PRESENCE_PENALTY
        assert result.frequency_penalty == ParameterValidator.DEFAULT_FREQUENCY_PENALTY

    def test_validate_request_empty_messages(self):
        """Test validation with empty messages."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        # Manually set empty messages to bypass Pydantic validation
        request.messages = []
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "Messages array cannot be empty" in str(exc_info.value)

    def test_validate_request_invalid_temperature(self):
        """Test validation with invalid temperature."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7
        )
        # Manually set invalid temperature to bypass Pydantic validation
        request.temperature = 3.0
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "Temperature must be between" in str(exc_info.value)

    def test_validate_request_invalid_top_p(self):
        """Test validation with invalid top_p."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            top_p=0.9
        )
        # Manually set invalid top_p to bypass Pydantic validation
        request.top_p = 1.5
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "top_p must be between" in str(exc_info.value)

    def test_validate_request_invalid_max_tokens(self):
        """Test validation with invalid max_tokens."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=5000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "max_tokens must be between" in str(exc_info.value)

    def test_validate_request_invalid_presence_penalty(self):
        """Test validation with invalid presence_penalty."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            presence_penalty=0.0
        )
        # Manually set invalid presence_penalty to bypass Pydantic validation
        request.presence_penalty = 3.0
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "presence_penalty must be between" in str(exc_info.value)

    def test_validate_request_invalid_frequency_penalty(self):
        """Test validation with invalid frequency_penalty."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            frequency_penalty=0.0
        )
        # Manually set invalid frequency_penalty to bypass Pydantic validation
        request.frequency_penalty = -3.0
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "frequency_penalty must be between" in str(exc_info.value)

    def test_validate_messages_too_many(self):
        """Test validation with too many messages."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(ParameterValidator.MAX_MESSAGE_COUNT + 1)
        ]
        
        self.validator._validate_messages(messages)
        
        assert any("Too many messages" in error for error in self.validator.validation_errors)

    def test_validate_message_content_too_long(self):
        """Test validation with message content too long."""
        long_content = "x" * (ParameterValidator.MAX_MESSAGE_LENGTH + 1)
        message = Message(role="user", content=long_content)
        
        self.validator._validate_message(message, 0)
        
        assert any("Content too long" in error for error in self.validator.validation_errors)

    def test_validate_message_empty_content(self):
        """Test validation with empty message content."""
        message = Message(role="user", content="Hello")
        # Manually set empty content to bypass Pydantic validation
        message.content = "   "
        
        self.validator._validate_message(message, 0)
        
        assert any("Content cannot be empty" in error for error in self.validator.validation_errors)

    def test_validate_message_invalid_role(self):
        """Test validation with invalid message role."""
        message = Message(role="user", content="Hello")
        # Manually set invalid role to bypass Pydantic validation
        message.role = "invalid"
        
        self.validator._validate_message(message, 0)
        
        assert any("Invalid role" in error for error in self.validator.validation_errors)

    def test_validate_message_multimodal_content(self):
        """Test validation with multimodal message content."""
        content_parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="image_url", image_url=ImageUrl(url="https://example.com/image.jpg"))
        ]
        message = Message(role="user", content=content_parts)
        
        self.validator._validate_message(message, 0)
        
        # Should not have validation errors for valid multimodal content
        assert not self.validator.validation_errors

    def test_validate_message_empty_image_url(self):
        """Test validation with empty image URL."""
        content_parts = [
            ContentPart(type="image_url", image_url=ImageUrl(url=""))
        ]
        message = Message(role="user", content=content_parts)
        
        self.validator._validate_message(message, 0)
        
        assert any("Image URL cannot be empty" in error for error in self.validator.validation_errors)

    def test_validate_message_sequence_system_messages(self):
        """Test validation of message sequence with system messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="You are a helpful assistant"),  # System message not at beginning
            Message(role="assistant", content="Hi there!")
        ]
        
        self.validator._validate_message_sequence(messages)
        
        assert any("System message" in error and "beginning" in error for error in self.validator.validation_errors)

    def test_normalize_stop_sequences_string(self):
        """Test normalizing stop sequences from string."""
        result = self.validator._normalize_stop_sequences("END")
        assert result == ["END"]

    def test_normalize_stop_sequences_list(self):
        """Test normalizing stop sequences from list."""
        result = self.validator._normalize_stop_sequences(["END", "STOP", ""])
        assert result == ["END", "STOP"]  # Empty string filtered out

    def test_normalize_stop_sequences_invalid(self):
        """Test normalizing invalid stop sequences."""
        result = self.validator._normalize_stop_sequences(123)
        assert result == []

    def test_validate_stop_sequences_too_many(self):
        """Test validation with too many stop sequences."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stop=["1", "2", "3", "4"]
        )
        # Manually set too many stop sequences to bypass Pydantic validation
        request.stop = ["1", "2", "3", "4", "5"]
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_request(request)
        
        assert "Too many stop sequences" in str(exc_info.value)

    def test_validate_model_compatibility(self):
        """Test model compatibility validation."""
        request = ChatCompletionRequest(
            model="different-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        result = self.validator.validate_model_compatibility(request, "available-model")
        
        assert result.model == "available-model"


class TestRequestValidator:
    """Test RequestValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = RequestValidator(available_model="test-model")

    def test_init(self):
        """Test RequestValidator initialization."""
        assert self.validator.available_model == "test-model"
        assert isinstance(self.validator.parameter_validator, ParameterValidator)

    def test_validate_and_normalize_success(self):
        """Test successful validation and normalization."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.8
        )
        
        result = self.validator.validate_and_normalize(request)
        
        assert result.model == "test-model"
        assert result.temperature == 0.8
        assert result.max_tokens == ParameterValidator.DEFAULT_MAX_TOKENS

    def test_validate_and_normalize_model_mismatch(self):
        """Test validation with model name mismatch."""
        request = ChatCompletionRequest(
            model="different-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        result = self.validator.validate_and_normalize(request)
        
        assert result.model == "test-model"  # Should be adjusted

    def test_validate_and_normalize_validation_error(self):
        """Test validation with validation error."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        # Manually set empty messages to bypass Pydantic validation
        request.messages = []
        
        with pytest.raises(ValidationError):
            self.validator.validate_and_normalize(request)

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        info = self.validator.get_parameter_info()
        
        assert "constraints" in info
        assert "limits" in info
        assert "available_model" in info
        assert info["available_model"] == "test-model"
        
        # Check temperature constraints
        temp_constraints = info["constraints"]["temperature"]
        assert temp_constraints["min"] == ParameterValidator.TEMPERATURE_MIN
        assert temp_constraints["max"] == ParameterValidator.TEMPERATURE_MAX
        assert temp_constraints["default"] == ParameterValidator.DEFAULT_TEMPERATURE


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_request_validator_default(self):
        """Test creating RequestValidator with default model."""
        validator = create_request_validator()
        
        assert isinstance(validator, RequestValidator)
        assert validator.available_model == "skt/A.X-4.0-VL-Light"

    def test_create_request_validator_custom(self):
        """Test creating RequestValidator with custom model."""
        validator = create_request_validator("custom-model")
        
        assert isinstance(validator, RequestValidator)
        assert validator.available_model == "custom-model"

    def test_validate_chat_request_success(self):
        """Test validating chat request successfully."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        result = validate_chat_request(request, "test-model")
        
        assert isinstance(result, ChatCompletionRequest)
        assert result.model == "test-model"

    def test_validate_chat_request_error(self):
        """Test validating chat request with error."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        # Manually set empty messages to bypass Pydantic validation
        request.messages = []
        
        with pytest.raises(ValidationError):
            validate_chat_request(request, "test-model")


class TestParameterValidatorIntegration:
    """Integration tests for ParameterValidator."""

    def test_comprehensive_validation(self):
        """Test comprehensive validation with multiple parameters."""
        validator = ParameterValidator()
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="system", content="You are a helpful assistant"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?")
            ],
            temperature=0.8,
            max_tokens=150,
            top_p=0.95,
            presence_penalty=0.1,
            frequency_penalty=-0.1,
            stop=["END", "STOP"]
        )
        
        result = validator.validate_request(request)
        
        # All parameters should be preserved
        assert result.temperature == 0.8
        assert result.max_tokens == 150
        assert result.top_p == 0.95
        assert result.presence_penalty == 0.1
        assert result.frequency_penalty == -0.1
        assert result.stop == ["END", "STOP"]

    def test_edge_case_parameters(self):
        """Test validation with edge case parameter values."""
        validator = ParameterValidator()
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,  # Minimum temperature
            max_tokens=1,     # Minimum max_tokens
            top_p=0.01,       # Near minimum top_p
            presence_penalty=-2.0,   # Minimum presence_penalty
            frequency_penalty=2.0    # Maximum frequency_penalty
        )
        
        result = validator.validate_request(request)
        
        # All edge case values should be accepted
        assert result.temperature == 0.0
        assert result.max_tokens == 1
        assert result.top_p == 0.01
        assert result.presence_penalty == -2.0
        assert result.frequency_penalty == 2.0

    def test_multiple_validation_errors(self):
        """Test validation with multiple errors."""
        validator = ParameterValidator()
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        # Manually set invalid values to bypass Pydantic validation
        request.messages = []
        request.temperature = 3.0
        request.max_tokens = -10
        request.top_p = 1.5
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_request(request)
        
        error_message = str(exc_info.value)
        assert "Messages array cannot be empty" in error_message
        assert "Temperature must be between" in error_message
        assert "max_tokens must be between" in error_message
        assert "top_p must be between" in error_message