"""Message processing utilities for converting OpenAI messages to model input format."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from app.schemas.chat_models import Message, ContentPart
from app.schemas.error_models import ValidationError

if TYPE_CHECKING:
    from app.services.model_manager import ModelManager


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
    
    def __init__(self, model_manager: Optional["ModelManager"] = None):
        """Initialize MessageProcessor.
        
        Args:
            model_manager: ModelManager instance for chat template support
        """
        self.model_manager = model_manager
        self.context = ConversationContext()
        
        # Fallback formatting templates (used when no chat template available)
        self.templates = {
            "system_prefix": "<|system|>\n",
            "user_prefix": "<|user|>\n", 
            "assistant_prefix": "<|assistant|>\n",
            "conversation_separator": "\n",
            "turn_separator": "\n",
        }
    
    def process_messages(
        self, 
        messages: List[Message], 
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Process OpenAI messages using official A.X-4.0-VL-Light format.
        
        Args:
            messages: List of OpenAI Message objects
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary containing processed model inputs and metadata
            
        Raises:
            ValidationError: If message format is invalid
        """
        if not messages:
            raise ValidationError("Messages list cannot be empty")
        
        try:
            # Convert OpenAI messages to official format
            processed_messages = []
            images = []
            has_images = False
            
            for message in messages:
                ax_content = []
                message_images = []
                
                if isinstance(message.content, str):
                    # Simple text message
                    ax_content.append({"type": "text", "text": message.content})
                    
                elif isinstance(message.content, list):
                    # Multi-modal content
                    for part in message.content:
                        if hasattr(part, 'type'):
                            if part.type == "text":
                                ax_content.append({"type": "text", "text": part.text or ""})
                            elif part.type == "image_url":
                                ax_content.append({"type": "image"})
                                image_url = part.image_url.url
                                message_images.append(image_url)
                                has_images = True
                                logger.info(f"Found image in message. URL length: {len(image_url)}, starts with: {image_url[:50]}...")
                
                # Add processed message
                if ax_content:  # Only add if there's content
                    processed_messages.append({
                        "role": message.role,
                        "content": ax_content
                    })
                
                # Store images for processor
                images.extend(message_images)
            
            # Use ModelManager's prepare_inputs method (official approach)
            if self.model_manager:
                try:
                    logger.debug(f"Calling prepare_inputs with processed_messages: {processed_messages}")
                    logger.debug(f"Images to process: {images if has_images else None}")
                    model_inputs = self.model_manager.prepare_inputs(
                        messages=processed_messages,
                        images=images if has_images else None
                    )
                    logger.debug("Using ModelManager prepare_inputs (official method)")
                    
                    return {
                        "model_inputs": model_inputs,
                        "has_images": has_images,
                        "images": images,
                        "message_count": len(processed_messages),
                        "processed_messages": processed_messages
                    }
                    
                except Exception as e:
                    logger.warning(f"ModelManager prepare_inputs failed: {e}. Using fallback.")
            
            # Fallback: convert to simple format and use fallback formatting
            simple_messages = []
            for msg in processed_messages:
                text_parts = [part["text"] for part in msg["content"] if part.get("type") == "text"]
                if text_parts:
                    simple_messages.append({
                        "role": msg["role"],
                        "content": " ".join(text_parts)
                    })
            
            formatted_text = self._fallback_format_messages(simple_messages)
            
            return {
                "text": formatted_text,
                "has_images": has_images,
                "images": images,
                "message_count": len(simple_messages),
                "processed_messages": processed_messages,
                "formatted_text": formatted_text  # For ResponseGenerator fallback
            }
            
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            raise ValidationError(f"Failed to process messages: {str(e)}")
    
    def _fallback_format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Fallback method to format messages when no chat template is available.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted chat string
        """
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"{self.templates['system_prefix']}{content}")
            elif role == "user":
                formatted_parts.append(f"{self.templates['user_prefix']}{content}")
            elif role == "assistant":
                formatted_parts.append(f"{self.templates['assistant_prefix']}{content}")
        
        # Add final assistant prompt for generation
        result = self.templates["conversation_separator"].join(formatted_parts)
        result += f"{self.templates['conversation_separator']}{self.templates['assistant_prefix']}"
        
        return result
    
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
    ) -> Dict[str, Any]:
        """Format model input for text generation.
        
        Args:
            model_input: Processed model input from process_messages
            add_assistant_prefix: Whether to add assistant prefix for generation
            
        Returns:
            Model inputs ready for generation (either tensor inputs or formatted text)
        """
        # If we have model_inputs from processor, return them directly
        if "model_inputs" in model_input:
            return model_input["model_inputs"]
        
        # Fallback: return formatted text
        return {"formatted_text": model_input.get("text", "")}
    
    def extract_assistant_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract assistant response from generated text.
        
        Args:
            generated_text: Full generated text from model
            original_prompt: Original prompt sent to model
            
        Returns:
            Extracted assistant response
        """
        response = generated_text.strip()
        
        # Remove the original prompt if it's at the beginning
        if response.startswith(original_prompt):
            response = response[len(original_prompt):].strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        # If response is empty or seems corrupted, try alternative extraction
        if not response or len(response.split()) < 2:
            response = self._alternative_extraction(generated_text, original_prompt)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        if not response:
            return ""
        
        response = response.strip()
        
        # Remove various assistant prefixes
        prefixes_to_remove = [
            self.templates["assistant_prefix"],
            "Assistant:",
            "Assistant: ",
            "<|assistant|>",
            "<|assistant|>\n"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Remove trailing conversation markers
        stop_markers = [
            self.templates["user_prefix"],
            self.templates["system_prefix"], 
            "Human:",
            "<|user|>",
            "<|system|>",
            "\nHuman:",
            "\n<|user|>",
            "\n<|system|>"
        ]
        
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        return response
    
    def _alternative_extraction(self, generated_text: str, original_prompt: str) -> str:
        """Alternative method to extract response when normal extraction fails.
        
        Args:
            generated_text: Full generated text
            original_prompt: Original prompt
            
        Returns:
            Extracted response or fallback message
        """
        # Try to find the last occurrence of assistant marker
        assistant_markers = ["Assistant:", "<|assistant|>", "assistant:", "Assistant: "]
        
        for marker in assistant_markers:
            last_idx = generated_text.rfind(marker)
            if last_idx >= 0:
                response = generated_text[last_idx + len(marker):].strip()
                if response and len(response.split()) >= 2:
                    return self._clean_response(response)
        
        # If nothing works, return a clean version of the whole text
        cleaned = self._clean_response(generated_text)
        if cleaned:
            return cleaned
        
        # Last resort
        return "죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다."
    
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