"""Message processing utilities for converting OpenAI messages to model input format."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from app.schemas.chat_models import Message, ContentPart
from app.schemas.error_models import ValidationError


logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages conversation history and context for model input."""
    
    def __init__(self, max_history_length: int = 10):
        """Initialize conversation context.
        
        Args:
            max_history_length: Maximum number of message pairs to keep in history
        """
        self.max_history_length = max_history_length
        self.system_message: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_tokens: int = 0
    
    def add_message(self, role: str, content: str, tokens: int = 0) -> None:
        """Add a message to conversation history.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
            tokens: Estimated token count for the message
        """
        if role == "system":
            self.system_message = content
        else:
            self.conversation_history.append({
                "role": role,
                "content": content,
                "tokens": tokens
            })
            self.total_tokens += tokens
            
            # Maintain history length limit (keep pairs of messages)
            while len(self.conversation_history) > self.max_history_length * 2:  # user + assistant pairs
                removed = self.conversation_history.pop(0)
                self.total_tokens -= removed.get("tokens", 0)
    
    def get_context_string(self) -> str:
        """Get formatted conversation context for model input."""
        context_parts = []
        
        # Add system message if present
        if self.system_message:
            context_parts.append(f"System: {self.system_message}")
        
        # Add conversation history
        for msg in self.conversation_history:
            role_prefix = {
                "user": "Human",
                "assistant": "Assistant"
            }.get(msg["role"], msg["role"].title())
            context_parts.append(f"{role_prefix}: {msg['content']}")
        
        return "\n\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.system_message = None
        self.conversation_history.clear()
        self.total_tokens = 0


class MessageProcessor:
    """Processes OpenAI messages and converts them to model input format."""
    
    def __init__(self, model_name: str = "skt/A.X-4.0-VL-Light"):
        """Initialize MessageProcessor.
        
        Args:
            model_name: Name of the target model for format optimization
        """
        self.model_name = model_name
        self.context = ConversationContext()
        
        # Model-specific formatting templates
        self.templates = {
            "system_prefix": "System: ",
            "user_prefix": "Human: ",
            "assistant_prefix": "Assistant: ",
            "conversation_separator": "\n\n",
            "turn_separator": "\n",
        }
    
    def process_messages(
        self, 
        messages: List[Message], 
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Process OpenAI messages into model input format.
        
        Args:
            messages: List of OpenAI Message objects
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary containing processed input for the model
            
        Raises:
            ValidationError: If message format is invalid
        """
        if not messages:
            raise ValidationError("Messages list cannot be empty")
        
        try:
            # Separate messages by role
            system_messages = []
            conversation_messages = []
            
            for message in messages:
                if message.role == "system":
                    system_messages.append(message)
                else:
                    conversation_messages.append(message)
            
            # Process system messages
            system_content = self._process_system_messages(system_messages)
            
            # Process conversation messages
            processed_messages = []
            has_images = False
            
            for message in conversation_messages:
                processed_msg = self._process_single_message(message)
                processed_messages.append(processed_msg)
                
                if processed_msg.get("has_image", False):
                    has_images = True
            
            # Build model input
            model_input = self._build_model_input(
                system_content=system_content,
                messages=processed_messages,
                include_history=include_history,
                has_images=has_images
            )
            
            # Update conversation context
            if include_history:
                self._update_context(system_content, processed_messages)
            
            return model_input
            
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            raise ValidationError(f"Failed to process messages: {str(e)}")
    
    def _process_system_messages(self, system_messages: List[Message]) -> Optional[str]:
        """Process system messages into a single system prompt.
        
        Args:
            system_messages: List of system messages
            
        Returns:
            Combined system message content or None
        """
        if not system_messages:
            return None
        
        system_contents = []
        for msg in system_messages:
            content = self._extract_text_content(msg.content)
            if content:
                system_contents.append(content)
        
        return " ".join(system_contents) if system_contents else None
    
    def _process_single_message(self, message: Message) -> Dict[str, Any]:
        """Process a single message into model format.
        
        Args:
            message: OpenAI Message object
            
        Returns:
            Processed message dictionary
        """
        processed = {
            "role": message.role,
            "has_image": False,
            "text_content": "",
            "image_data": None,
            "name": message.name
        }
        
        if isinstance(message.content, str):
            # Simple text message
            processed["text_content"] = message.content
        elif isinstance(message.content, list):
            # Multimodal message with text and/or images
            text_parts = []
            image_data = None
            
            for part in message.content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    processed["has_image"] = True
                    # Store image URL/data for later processing
                    image_data = {
                        "url": part.image_url.url,
                        "detail": part.image_url.detail or "auto"
                    }
            
            processed["text_content"] = " ".join(text_parts)
            processed["image_data"] = image_data
        
        return processed
    
    def _extract_text_content(self, content: Union[str, List[ContentPart]]) -> str:
        """Extract text content from message content.
        
        Args:
            content: Message content (string or list of content parts)
            
        Returns:
            Extracted text content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
            return " ".join(text_parts)
        return ""
    
    def _build_model_input(
        self,
        system_content: Optional[str],
        messages: List[Dict[str, Any]],
        include_history: bool,
        has_images: bool
    ) -> Dict[str, Any]:
        """Build final model input format.
        
        Args:
            system_content: System message content
            messages: Processed conversation messages
            include_history: Whether to include conversation history
            has_images: Whether the input contains images
            
        Returns:
            Model input dictionary
        """
        # Build conversation text
        conversation_parts = []
        
        # Add system message
        if system_content:
            conversation_parts.append(f"{self.templates['system_prefix']}{system_content}")
        
        # Add conversation history if requested
        if include_history and self.context.conversation_history:
            history_context = self.context.get_context_string()
            if history_context and not system_content:
                conversation_parts.append(history_context)
            elif history_context:
                # Extract only the conversation part (skip system message from history)
                history_lines = history_context.split('\n\n')
                conversation_history = [line for line in history_lines if not line.startswith('System:')]
                if conversation_history:
                    conversation_parts.extend(conversation_history)
        
        # Add current messages
        for msg in messages:
            role_prefix = {
                "user": self.templates["user_prefix"],
                "assistant": self.templates["assistant_prefix"]
            }.get(msg["role"], f"{msg['role'].title()}: ")
            
            conversation_parts.append(f"{role_prefix}{msg['text_content']}")
        
        # Build final input
        conversation_text = self.templates["conversation_separator"].join(conversation_parts)
        
        # Prepare model input
        model_input = {
            "text": conversation_text,
            "has_images": has_images,
            "conversation_length": len(messages),
            "system_message": system_content,
            "format_version": "v1"
        }
        
        # Add image data if present
        if has_images:
            image_data = []
            for msg in messages:
                if msg.get("image_data"):
                    image_data.append(msg["image_data"])
            model_input["images"] = image_data
        
        return model_input
    
    def _update_context(
        self, 
        system_content: Optional[str], 
        messages: List[Dict[str, Any]]
    ) -> None:
        """Update conversation context with new messages.
        
        Args:
            system_content: System message content
            messages: Processed conversation messages
        """
        # Update system message
        if system_content:
            self.context.system_message = system_content
        
        # Add conversation messages to history
        for msg in messages:
            # Estimate token count (rough approximation)
            token_count = len(msg["text_content"].split()) * 1.3  # Rough token estimation
            
            self.context.add_message(
                role=msg["role"],
                content=msg["text_content"],
                tokens=int(token_count)
            )
    
    def format_for_generation(
        self, 
        model_input: Dict[str, Any], 
        add_assistant_prefix: bool = True
    ) -> str:
        """Format model input for text generation.
        
        Args:
            model_input: Processed model input
            add_assistant_prefix: Whether to add assistant prefix for generation
            
        Returns:
            Formatted text for model generation
        """
        text = model_input["text"]
        
        if add_assistant_prefix:
            # Add assistant prefix to prompt generation
            if not text.endswith(self.templates["assistant_prefix"]):
                text += f"{self.templates['conversation_separator']}{self.templates['assistant_prefix']}"
        
        return text
    
    def extract_assistant_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract assistant response from generated text.
        
        Args:
            generated_text: Full generated text from model
            original_prompt: Original prompt sent to model
            
        Returns:
            Extracted assistant response
        """
        # Remove the original prompt from generated text
        if generated_text.startswith(original_prompt):
            response = generated_text[len(original_prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove common artifacts
        response = response.strip()
        
        # Remove assistant prefix if it appears at the start
        if response.startswith(self.templates["assistant_prefix"]):
            response = response[len(self.templates["assistant_prefix"]):].strip()
        
        # Remove any trailing conversation markers
        for prefix in [self.templates["user_prefix"], self.templates["system_prefix"]]:
            if prefix in response:
                response = response.split(prefix)[0].strip()
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        return response
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            "total_messages": len(self.context.conversation_history),
            "total_tokens": self.context.total_tokens,
            "has_system_message": self.context.system_message is not None,
            "max_history_length": self.context.max_history_length
        }
    
    def reset_context(self) -> None:
        """Reset conversation context."""
        self.context.clear()
    
    def set_templates(self, templates: Dict[str, str]) -> None:
        """Update formatting templates.
        
        Args:
            templates: Dictionary of template strings
        """
        self.templates.update(templates)


# Utility functions for message processing
def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~1.3 tokens per word for English text
    word_count = len(text.split())
    return int(word_count * 1.3)


def validate_message_sequence(messages: List[Message]) -> bool:
    """Validate that message sequence follows proper conversation flow.
    
    Args:
        messages: List of messages to validate
        
    Returns:
        True if sequence is valid
    """
    if not messages:
        return False
    
    # Check for alternating user/assistant pattern (after system messages)
    conversation_roles = []
    for msg in messages:
        if msg.role != "system":
            conversation_roles.append(msg.role)
    
    if not conversation_roles:
        return False
    
    # Should start with user message
    if conversation_roles[0] != "user":
        return False
    
    # Check alternating pattern
    for i in range(1, len(conversation_roles)):
        prev_role = conversation_roles[i-1]
        curr_role = conversation_roles[i]
        
        # Should alternate between user and assistant
        if prev_role == "user" and curr_role != "assistant":
            return False
        elif prev_role == "assistant" and curr_role != "user":
            return False
    
    return True