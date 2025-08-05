"""Request validation and parameter handling for chat completion requests."""

import logging
from typing import Dict, Any, List, Optional, Union

from app.schemas.chat_models import ChatCompletionRequest, Message
from app.schemas.error_models import ValidationError


logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates and normalizes request parameters."""
    
    # Parameter constraints
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 2.0
    TOP_P_MIN = 0.0
    TOP_P_MAX = 1.0
    MAX_TOKENS_MIN = 1
    MAX_TOKENS_MAX = 4096
    PRESENCE_PENALTY_MIN = -2.0
    PRESENCE_PENALTY_MAX = 2.0
    FREQUENCY_PENALTY_MIN = -2.0
    FREQUENCY_PENALTY_MAX = 2.0
    MAX_STOP_SEQUENCES = 4
    MAX_MESSAGE_COUNT = 100
    MAX_MESSAGE_LENGTH = 10000
    
    # Default values
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 1.0
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_FREQUENCY_PENALTY = 0.0
    
    def __init__(self):
        """Initialize ParameterValidator."""
        self.validation_errors = []
    
    def validate_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Validate and normalize a chat completion request.
        
        Args:
            request: ChatCompletionRequest to validate
            
        Returns:
            Validated and normalized request
            
        Raises:
            ValidationError: If validation fails
        """
        self.validation_errors = []
        
        # Validate messages
        self._validate_messages(request.messages)
        
        # Validate and normalize parameters
        request = self._validate_and_normalize_parameters(request)
        
        # Check for validation errors
        if self.validation_errors:
            error_message = "; ".join(self.validation_errors)
            raise ValidationError(f"Request validation failed: {error_message}")
        
        return request
    
    def _validate_messages(self, messages: List[Message]):
        """Validate messages array.
        
        Args:
            messages: List of messages to validate
        """
        if not messages:
            self.validation_errors.append("Messages array cannot be empty")
            return
        
        if len(messages) > self.MAX_MESSAGE_COUNT:
            self.validation_errors.append(
                f"Too many messages: {len(messages)} (max: {self.MAX_MESSAGE_COUNT})"
            )
        
        # Validate individual messages
        for i, message in enumerate(messages):
            self._validate_message(message, i)
        
        # Validate message sequence
        self._validate_message_sequence(messages)
    
    def _validate_message(self, message: Message, index: int):
        """Validate individual message.
        
        Args:
            message: Message to validate
            index: Message index for error reporting
        """
        # Validate role
        if message.role not in ["system", "user", "assistant"]:
            self.validation_errors.append(
                f"Message {index}: Invalid role '{message.role}'"
            )
        
        # Validate content
        if isinstance(message.content, str):
            if len(message.content) > self.MAX_MESSAGE_LENGTH:
                self.validation_errors.append(
                    f"Message {index}: Content too long ({len(message.content)} chars, max: {self.MAX_MESSAGE_LENGTH})"
                )
            if not message.content.strip():
                self.validation_errors.append(
                    f"Message {index}: Content cannot be empty"
                )
        elif isinstance(message.content, list):
            if not message.content:
                self.validation_errors.append(
                    f"Message {index}: Content array cannot be empty"
                )
            
            # Validate content parts
            for j, part in enumerate(message.content):
                if part.type == "text" and part.text:
                    if len(part.text) > self.MAX_MESSAGE_LENGTH:
                        self.validation_errors.append(
                            f"Message {index}, part {j}: Text too long"
                        )
                elif part.type == "image_url" and part.image_url:
                    if not part.image_url.url:
                        self.validation_errors.append(
                            f"Message {index}, part {j}: Image URL cannot be empty"
                        )
    
    def _validate_message_sequence(self, messages: List[Message]):
        """Validate message sequence follows conversation rules.
        
        Args:
            messages: List of messages to validate
        """
        if not messages:
            return
        
        # Check for consecutive messages with same role (except system)
        prev_role = None
        for i, message in enumerate(messages):
            if message.role == "system":
                # System messages should be at the beginning
                if i > 0 and any(m.role != "system" for m in messages[:i]):
                    self.validation_errors.append(
                        f"System message at position {i} should be at the beginning"
                    )
            elif message.role == prev_role and prev_role != "system":
                logger.warning(
                    f"Consecutive messages with same role '{message.role}' at position {i}"
                )
            
            prev_role = message.role
    
    def _validate_and_normalize_parameters(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """Validate and normalize request parameters.
        
        Args:
            request: Request to validate and normalize
            
        Returns:
            Normalized request
        """
        # Temperature validation and normalization
        if request.temperature is not None:
            if not (self.TEMPERATURE_MIN <= request.temperature <= self.TEMPERATURE_MAX):
                self.validation_errors.append(
                    f"Temperature must be between {self.TEMPERATURE_MIN} and {self.TEMPERATURE_MAX}, got {request.temperature}"
                )
        else:
            request.temperature = self.DEFAULT_TEMPERATURE
        
        # Top-p validation and normalization
        if request.top_p is not None:
            if not (self.TOP_P_MIN < request.top_p <= self.TOP_P_MAX):
                self.validation_errors.append(
                    f"top_p must be between {self.TOP_P_MIN} (exclusive) and {self.TOP_P_MAX}, got {request.top_p}"
                )
        else:
            request.top_p = self.DEFAULT_TOP_P
        
        # Max tokens validation and normalization
        if request.max_tokens is not None:
            if not (self.MAX_TOKENS_MIN <= request.max_tokens <= self.MAX_TOKENS_MAX):
                self.validation_errors.append(
                    f"max_tokens must be between {self.MAX_TOKENS_MIN} and {self.MAX_TOKENS_MAX}, got {request.max_tokens}"
                )
        else:
            request.max_tokens = self.DEFAULT_MAX_TOKENS
        
        # Presence penalty validation and normalization
        if request.presence_penalty is not None:
            if not (self.PRESENCE_PENALTY_MIN <= request.presence_penalty <= self.PRESENCE_PENALTY_MAX):
                self.validation_errors.append(
                    f"presence_penalty must be between {self.PRESENCE_PENALTY_MIN} and {self.PRESENCE_PENALTY_MAX}, got {request.presence_penalty}"
                )
        else:
            request.presence_penalty = self.DEFAULT_PRESENCE_PENALTY
        
        # Frequency penalty validation and normalization
        if request.frequency_penalty is not None:
            if not (self.FREQUENCY_PENALTY_MIN <= request.frequency_penalty <= self.FREQUENCY_PENALTY_MAX):
                self.validation_errors.append(
                    f"frequency_penalty must be between {self.FREQUENCY_PENALTY_MIN} and {self.FREQUENCY_PENALTY_MAX}, got {request.frequency_penalty}"
                )
        else:
            request.frequency_penalty = self.DEFAULT_FREQUENCY_PENALTY
        
        # Stop sequences validation
        if request.stop is not None:
            stop_sequences = self._normalize_stop_sequences(request.stop)
            if len(stop_sequences) > self.MAX_STOP_SEQUENCES:
                self.validation_errors.append(
                    f"Too many stop sequences: {len(stop_sequences)} (max: {self.MAX_STOP_SEQUENCES})"
                )
            request.stop = stop_sequences
        
        # Stream parameter validation
        if request.stream is None:
            request.stream = False
        
        return request
    
    def _normalize_stop_sequences(self, stop: Union[str, List[str]]) -> List[str]:
        """Normalize stop sequences to list format.
        
        Args:
            stop: Stop sequences (string or list)
            
        Returns:
            List of stop sequences
        """
        if isinstance(stop, str):
            return [stop] if stop.strip() else []
        elif isinstance(stop, list):
            return [s for s in stop if isinstance(s, str) and s.strip()]
        else:
            return []
    
    def validate_model_compatibility(
        self, 
        request: ChatCompletionRequest, 
        available_model: str
    ) -> ChatCompletionRequest:
        """Validate model compatibility and adjust request if needed.
        
        Args:
            request: Request to validate
            available_model: Available model name
            
        Returns:
            Adjusted request
        """
        if request.model != available_model:
            logger.warning(
                f"Requested model '{request.model}' not available, using '{available_model}'"
            )
            request.model = available_model
        
        return request


class RequestValidator:
    """Main request validator that orchestrates validation process."""
    
    def __init__(self, available_model: str = "skt/A.X-4.0-VL-Light"):
        """Initialize RequestValidator.
        
        Args:
            available_model: Name of the available model
        """
        self.available_model = available_model
        self.parameter_validator = ParameterValidator()
    
    def validate_and_normalize(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """Validate and normalize a chat completion request.
        
        Args:
            request: Request to validate
            
        Returns:
            Validated and normalized request
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate basic request structure and parameters
            request = self.parameter_validator.validate_request(request)
            
            # Validate model compatibility
            request = self.parameter_validator.validate_model_compatibility(
                request, self.available_model
            )
            
            logger.debug(f"Request validated successfully: {request.model}")
            return request
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about parameter constraints and defaults.
        
        Returns:
            Dictionary with parameter information
        """
        return {
            "constraints": {
                "temperature": {
                    "min": ParameterValidator.TEMPERATURE_MIN,
                    "max": ParameterValidator.TEMPERATURE_MAX,
                    "default": ParameterValidator.DEFAULT_TEMPERATURE
                },
                "top_p": {
                    "min": ParameterValidator.TOP_P_MIN,
                    "max": ParameterValidator.TOP_P_MAX,
                    "default": ParameterValidator.DEFAULT_TOP_P
                },
                "max_tokens": {
                    "min": ParameterValidator.MAX_TOKENS_MIN,
                    "max": ParameterValidator.MAX_TOKENS_MAX,
                    "default": ParameterValidator.DEFAULT_MAX_TOKENS
                },
                "presence_penalty": {
                    "min": ParameterValidator.PRESENCE_PENALTY_MIN,
                    "max": ParameterValidator.PRESENCE_PENALTY_MAX,
                    "default": ParameterValidator.DEFAULT_PRESENCE_PENALTY
                },
                "frequency_penalty": {
                    "min": ParameterValidator.FREQUENCY_PENALTY_MIN,
                    "max": ParameterValidator.FREQUENCY_PENALTY_MAX,
                    "default": ParameterValidator.DEFAULT_FREQUENCY_PENALTY
                }
            },
            "limits": {
                "max_stop_sequences": ParameterValidator.MAX_STOP_SEQUENCES,
                "max_message_count": ParameterValidator.MAX_MESSAGE_COUNT,
                "max_message_length": ParameterValidator.MAX_MESSAGE_LENGTH
            },
            "available_model": self.available_model
        }


# Utility functions
def create_request_validator(model_name: str = "skt/A.X-4.0-VL-Light") -> RequestValidator:
    """Create a RequestValidator instance.
    
    Args:
        model_name: Name of the available model
        
    Returns:
        RequestValidator instance
    """
    return RequestValidator(available_model=model_name)


def validate_chat_request(
    request: ChatCompletionRequest,
    model_name: str = "skt/A.X-4.0-VL-Light"
) -> ChatCompletionRequest:
    """Validate and normalize a chat completion request.
    
    Args:
        request: Request to validate
        model_name: Available model name
        
    Returns:
        Validated and normalized request
        
    Raises:
        ValidationError: If validation fails
    """
    validator = create_request_validator(model_name)
    return validator.validate_and_normalize(request)