"""Unit tests for MessageProcessor class."""

import pytest
from unittest.mock import Mock, patch
from app.models.message_processor import (
    MessageProcessor, 
    ConversationContext,
    estimate_token_count,
    validate_message_sequence
)
from app.schemas.chat_models import Message, ContentPart, ImageUrl
from app.schemas.error_models import ValidationError


class TestConversationContext:
    """Test ConversationContext class."""

    def test_init_default(self):
        """Test ConversationContext initialization with defaults."""
        context = ConversationContext()
        assert context.max_history_length == 10
        assert context.system_message is None
        assert context.conversation_history == []
        assert context.total_tokens == 0

    def test_init_custom(self):
        """Test ConversationContext initialization with custom values."""
        context = ConversationContext(max_history_length=5)
        assert context.max_history_length == 5

    def test_add_system_message(self):
        """Test adding system message."""
        context = ConversationContext()
        context.add_message("system", "You are a helpful assistant", 10)
        
        assert context.system_message == "You are a helpful assistant"
        assert len(context.conversation_history) == 0
        assert context.total_tokens == 0  # System messages don't count toward total

    def test_add_conversation_messages(self):
        """Test adding conversation messages."""
        context = ConversationContext()
        
        context.add_message("user", "Hello", 5)
        context.add_message("assistant", "Hi there!", 10)
        
        assert len(context.conversation_history) == 2
        assert context.total_tokens == 15
        assert context.conversation_history[0]["role"] == "user"
        assert context.conversation_history[1]["role"] == "assistant"

    def test_history_length_limit(self):
        """Test conversation history length limiting."""
        context = ConversationContext(max_history_length=2)
        
        # Add more messages than the limit
        for i in range(6):  # 3 user + 3 assistant = 6 messages
            context.add_message("user", f"Message {i}", 5)
            context.add_message("assistant", f"Response {i}", 5)
        
        # Should only keep the last 4 messages (2 pairs)
        assert len(context.conversation_history) == 4
        assert context.conversation_history[0]["content"] == "Message 4"  # Last 2 pairs start from Message 4
        assert context.conversation_history[2]["content"] == "Message 5"
        assert context.total_tokens == 20  # 4 messages * 5 tokens each

    def test_get_context_string_empty(self):
        """Test getting context string when empty."""
        context = ConversationContext()
        assert context.get_context_string() == ""

    def test_get_context_string_with_system(self):
        """Test getting context string with system message."""
        context = ConversationContext()
        context.add_message("system", "You are helpful", 0)
        context.add_message("user", "Hello", 5)
        context.add_message("assistant", "Hi!", 5)
        
        context_str = context.get_context_string()
        expected = "System: You are helpful\n\nHuman: Hello\n\nAssistant: Hi!"
        assert context_str == expected

    def test_get_context_string_no_system(self):
        """Test getting context string without system message."""
        context = ConversationContext()
        context.add_message("user", "Hello", 5)
        context.add_message("assistant", "Hi!", 5)
        
        context_str = context.get_context_string()
        expected = "Human: Hello\n\nAssistant: Hi!"
        assert context_str == expected

    def test_clear(self):
        """Test clearing conversation context."""
        context = ConversationContext()
        context.add_message("system", "System message", 0)
        context.add_message("user", "Hello", 5)
        context.add_message("assistant", "Hi!", 5)
        
        context.clear()
        
        assert context.system_message is None
        assert context.conversation_history == []
        assert context.total_tokens == 0


class TestMessageProcessor:
    """Test MessageProcessor class."""

    def test_init_default(self):
        """Test MessageProcessor initialization with defaults."""
        processor = MessageProcessor()
        assert processor.model_name == "skt/A.X-4.0-VL-Light"
        assert isinstance(processor.context, ConversationContext)
        assert "system_prefix" in processor.templates

    def test_init_custom_model(self):
        """Test MessageProcessor initialization with custom model."""
        processor = MessageProcessor(model_name="custom/model")
        assert processor.model_name == "custom/model"

    def test_process_messages_empty_list(self):
        """Test processing empty messages list."""
        processor = MessageProcessor()
        
        with pytest.raises(ValidationError) as exc_info:
            processor.process_messages([])
        assert "Messages list cannot be empty" in str(exc_info.value)

    def test_process_simple_text_message(self):
        """Test processing simple text message."""
        processor = MessageProcessor()
        messages = [Message(role="user", content="Hello, world!")]
        
        result = processor.process_messages(messages, include_history=False)
        
        assert result["text"] == "Human: Hello, world!"
        assert result["has_images"] is False
        assert result["conversation_length"] == 1
        assert result["system_message"] is None

    def test_process_system_message(self):
        """Test processing system message."""
        processor = MessageProcessor()
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello")
        ]
        
        result = processor.process_messages(messages, include_history=False)
        
        assert "System: You are a helpful assistant" in result["text"]
        assert "Human: Hello" in result["text"]
        assert result["system_message"] == "You are a helpful assistant"

    def test_process_multimodal_message(self):
        """Test processing multimodal message with text and image."""
        processor = MessageProcessor()
        
        content_parts = [
            ContentPart(type="text", text="What's in this image?"),
            ContentPart(
                type="image_url", 
                image_url=ImageUrl(url="https://example.com/image.jpg", detail="high")
            )
        ]
        messages = [Message(role="user", content=content_parts)]
        
        result = processor.process_messages(messages, include_history=False)
        
        assert result["has_images"] is True
        assert "What's in this image?" in result["text"]
        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["url"] == "https://example.com/image.jpg"
        assert result["images"][0]["detail"] == "high"

    def test_process_multiple_system_messages(self):
        """Test processing multiple system messages."""
        processor = MessageProcessor()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="system", content="Be concise."),
            Message(role="user", content="Hello")
        ]
        
        result = processor.process_messages(messages, include_history=False)
        
        assert result["system_message"] == "You are helpful. Be concise."
        assert "System: You are helpful. Be concise." in result["text"]

    def test_process_conversation_with_history(self):
        """Test processing conversation with history enabled."""
        processor = MessageProcessor()
        
        # Add some history first
        processor.context.add_message("user", "Previous question", 10)
        processor.context.add_message("assistant", "Previous answer", 10)
        
        messages = [Message(role="user", content="New question")]
        
        result = processor.process_messages(messages, include_history=True)
        
        # Should include previous conversation
        assert "Previous question" in result["text"] or "Human: Previous question" in result["text"]
        assert "New question" in result["text"]

    def test_extract_text_content_string(self):
        """Test extracting text content from string."""
        processor = MessageProcessor()
        content = "Hello, world!"
        
        result = processor._extract_text_content(content)
        assert result == "Hello, world!"

    def test_extract_text_content_list(self):
        """Test extracting text content from content parts list."""
        processor = MessageProcessor()
        content_parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="text", text="world"),
            ContentPart(
                type="image_url", 
                image_url=ImageUrl(url="https://example.com/image.jpg")
            )
        ]
        
        result = processor._extract_text_content(content_parts)
        assert result == "Hello world"

    def test_format_for_generation_with_prefix(self):
        """Test formatting for generation with assistant prefix."""
        processor = MessageProcessor()
        model_input = {"text": "Human: Hello"}
        
        result = processor.format_for_generation(model_input, add_assistant_prefix=True)
        
        assert result.endswith("Assistant: ")

    def test_format_for_generation_without_prefix(self):
        """Test formatting for generation without assistant prefix."""
        processor = MessageProcessor()
        model_input = {"text": "Human: Hello"}
        
        result = processor.format_for_generation(model_input, add_assistant_prefix=False)
        
        assert result == "Human: Hello"

    def test_extract_assistant_response(self):
        """Test extracting assistant response from generated text."""
        processor = MessageProcessor()
        original_prompt = "Human: Hello\n\nAssistant: "
        generated_text = "Human: Hello\n\nAssistant: Hi there! How can I help you?"
        
        result = processor.extract_assistant_response(generated_text, original_prompt)
        
        assert result == "Hi there! How can I help you?"

    def test_clean_response(self):
        """Test cleaning response text."""
        processor = MessageProcessor()
        
        # Test removing assistant prefix
        response = "Assistant: Hello there!"
        cleaned = processor._clean_response(response)
        assert cleaned == "Hello there!"
        
        # Test removing user prefix interruption
        response = "Hello there! Human: What about this?"
        cleaned = processor._clean_response(response)
        assert cleaned == "Hello there!"
        
        # Test whitespace cleanup
        response = "  Hello   there!  "
        cleaned = processor._clean_response(response)
        assert cleaned == "Hello there!"

    def test_get_conversation_stats(self):
        """Test getting conversation statistics."""
        processor = MessageProcessor()
        processor.context.add_message("system", "System message", 0)
        processor.context.add_message("user", "Hello", 5)
        processor.context.add_message("assistant", "Hi", 3)
        
        stats = processor.get_conversation_stats()
        
        assert stats["total_messages"] == 2  # Only conversation messages
        assert stats["total_tokens"] == 8
        assert stats["has_system_message"] is True
        assert stats["max_history_length"] == 10

    def test_reset_context(self):
        """Test resetting conversation context."""
        processor = MessageProcessor()
        processor.context.add_message("user", "Hello", 5)
        
        processor.reset_context()
        
        assert len(processor.context.conversation_history) == 0
        assert processor.context.total_tokens == 0

    def test_set_templates(self):
        """Test setting custom templates."""
        processor = MessageProcessor()
        custom_templates = {
            "user_prefix": "User: ",
            "assistant_prefix": "Bot: "
        }
        
        processor.set_templates(custom_templates)
        
        assert processor.templates["user_prefix"] == "User: "
        assert processor.templates["assistant_prefix"] == "Bot: "

    def test_process_message_with_name(self):
        """Test processing message with name field."""
        processor = MessageProcessor()
        messages = [Message(role="user", content="Hello", name="John")]
        
        result = processor.process_messages(messages, include_history=False)
        
        # Name should be preserved in the processing but not affect text output
        assert "Human: Hello" in result["text"]

    def test_process_base64_image(self):
        """Test processing message with base64 image."""
        processor = MessageProcessor()
        
        base64_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        content_parts = [
            ContentPart(type="text", text="Analyze this image"),
            ContentPart(
                type="image_url", 
                image_url=ImageUrl(url=base64_url)
            )
        ]
        messages = [Message(role="user", content=content_parts)]
        
        result = processor.process_messages(messages, include_history=False)
        
        assert result["has_images"] is True
        assert result["images"][0]["url"] == base64_url
        assert result["images"][0]["detail"] == "auto"  # Default value

    def test_error_handling_in_process_messages(self):
        """Test error handling in process_messages."""
        processor = MessageProcessor()
        
        # Create a message that will cause an error during processing
        with patch.object(processor, '_process_single_message', side_effect=Exception("Test error")):
            with pytest.raises(ValidationError) as exc_info:
                processor.process_messages([Message(role="user", content="Hello")])
            assert "Failed to process messages" in str(exc_info.value)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_token_count(self):
        """Test token count estimation."""
        # Test empty string
        assert estimate_token_count("") == 0
        
        # Test single word
        assert estimate_token_count("hello") == 1  # 1 * 1.3 = 1.3 -> 1
        
        # Test multiple words
        assert estimate_token_count("hello world test") == 3  # 3 * 1.3 = 3.9 -> 3
        
        # Test longer text
        text = "This is a longer sentence with multiple words"
        expected = int(len(text.split()) * 1.3)
        assert estimate_token_count(text) == expected

    def test_validate_message_sequence_empty(self):
        """Test validating empty message sequence."""
        assert validate_message_sequence([]) is False

    def test_validate_message_sequence_valid(self):
        """Test validating valid message sequence."""
        messages = [
            Message(role="system", content="System message"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm good")
        ]
        
        assert validate_message_sequence(messages) is True

    def test_validate_message_sequence_starts_with_assistant(self):
        """Test validating sequence that starts with assistant."""
        messages = [
            Message(role="assistant", content="Hi there!")
        ]
        
        assert validate_message_sequence(messages) is False

    def test_validate_message_sequence_invalid_alternation(self):
        """Test validating sequence with invalid alternation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="user", content="Another user message")  # Should be assistant
        ]
        
        assert validate_message_sequence(messages) is False

    def test_validate_message_sequence_only_system(self):
        """Test validating sequence with only system messages."""
        messages = [
            Message(role="system", content="System message 1"),
            Message(role="system", content="System message 2")
        ]
        
        assert validate_message_sequence(messages) is False

    def test_validate_message_sequence_system_then_valid(self):
        """Test validating sequence with system messages followed by valid conversation."""
        messages = [
            Message(role="system", content="System message"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi")
        ]
        
        assert validate_message_sequence(messages) is True


class TestMessageProcessorIntegration:
    """Integration tests for MessageProcessor."""

    def test_full_conversation_flow(self):
        """Test complete conversation processing flow."""
        processor = MessageProcessor()
        
        # Process initial system + user message
        messages1 = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello, how are you?")
        ]
        
        result1 = processor.process_messages(messages1)
        
        # Verify initial processing
        assert "System: You are a helpful assistant" in result1["text"]
        assert "Human: Hello, how are you?" in result1["text"]
        assert result1["has_images"] is False
        
        # Process follow-up message (should include history)
        messages2 = [
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="What can you help me with?")
        ]
        
        result2 = processor.process_messages(messages2)
        
        # Should include previous context
        assert "What can you help me with?" in result2["text"]
        
        # Check conversation stats
        stats = processor.get_conversation_stats()
        assert stats["total_messages"] > 0
        assert stats["has_system_message"] is True

    def test_multimodal_conversation_flow(self):
        """Test conversation flow with multimodal messages."""
        processor = MessageProcessor()
        
        # First message with image
        content_parts = [
            ContentPart(type="text", text="What's in this image?"),
            ContentPart(
                type="image_url", 
                image_url=ImageUrl(url="https://example.com/image.jpg", detail="high")
            )
        ]
        messages1 = [Message(role="user", content=content_parts)]
        
        result1 = processor.process_messages(messages1)
        
        assert result1["has_images"] is True
        assert len(result1["images"]) == 1
        
        # Follow-up text message
        messages2 = [
            Message(role="assistant", content="I can see a cat in the image."),
            Message(role="user", content="What color is the cat?")
        ]
        
        result2 = processor.process_messages(messages2)
        
        # Should not have images in second batch
        assert result2["has_images"] is False
        assert "images" not in result2 or len(result2.get("images", [])) == 0

    def test_generation_formatting_flow(self):
        """Test complete flow from processing to generation formatting."""
        processor = MessageProcessor()
        
        messages = [
            Message(role="system", content="Be helpful and concise"),
            Message(role="user", content="Explain quantum computing")
        ]
        
        # Process messages
        model_input = processor.process_messages(messages)
        
        # Format for generation
        generation_prompt = processor.format_for_generation(model_input)
        
        # Should end with assistant prefix
        assert generation_prompt.endswith("Assistant: ")
        
        # Simulate model generation
        generated_text = generation_prompt + "Quantum computing uses quantum mechanics principles..."
        
        # Extract response
        response = processor.extract_assistant_response(generated_text, generation_prompt)
        
        assert response == "Quantum computing uses quantum mechanics principles..."
        assert not response.startswith("Assistant: ")