"""Unit tests for ResponseGenerator and related classes."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
from transformers import GenerationConfig

from app.models.response_generator import (
    ResponseGenerator,
    TokenCounter,
    GenerationParameters,
    SSEFormatter,
    StreamingTokenizer,
    validate_generation_request,
    estimate_generation_time
)
from app.services.model_manager import ModelManager
from app.models.message_processor import MessageProcessor
from app.models.image_handler import ImageHandler
from app.schemas.chat_models import ChatCompletionRequest, Message, Usage
from app.schemas.error_models import ValidationError, ModelError


class TestSSEFormatter:
    """Test SSEFormatter class."""

    def test_format_chunk(self):
        """Test formatting data chunk for SSE."""
        data = {
            "id": "test-123",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "Hello"}}]
        }
        
        result = SSEFormatter.format_chunk(data)
        
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert "test-123" in result
        assert "Hello" in result

    def test_format_done(self):
        """Test formatting [DONE] message."""
        result = SSEFormatter.format_done()
        
        assert result == "data: [DONE]\n\n"

    def test_format_error(self):
        """Test formatting error message."""
        error_msg = "Something went wrong"
        
        result = SSEFormatter.format_error(error_msg)
        
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert error_msg in result
        assert "server_error" in result


class TestStreamingTokenizer:
    """Test StreamingTokenizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        self.streaming_tokenizer = StreamingTokenizer(
            self.mock_tokenizer,
            self.mock_model
        )

    def test_init(self):
        """Test StreamingTokenizer initialization."""
        assert self.streaming_tokenizer.tokenizer == self.mock_tokenizer
        assert self.streaming_tokenizer.model == self.mock_model
        assert self.streaming_tokenizer.streamer is None

    def test_create_streamer(self):
        """Test creating TextIteratorStreamer."""
        with patch('app.models.response_generator.TextIteratorStreamer') as mock_streamer_class:
            mock_streamer = Mock()
            mock_streamer_class.return_value = mock_streamer
            
            result = self.streaming_tokenizer.create_streamer()
            
            assert result == mock_streamer
            assert self.streaming_tokenizer.streamer == mock_streamer
            mock_streamer_class.assert_called_once_with(
                self.mock_tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0
            )

    @pytest.mark.asyncio
    async def test_generate_streaming_success(self):
        """Test successful streaming generation."""
        # Mock streamer
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " ", "world"]))
        
        with patch.object(self.streaming_tokenizer, 'create_streamer', return_value=mock_streamer):
            with patch('app.models.response_generator.Thread') as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread
                
                inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
                gen_config = GenerationConfig()
                
                tokens = []
                async for token in self.streaming_tokenizer.generate_streaming(inputs, gen_config):
                    tokens.append(token)
                
                assert tokens == ["Hello", " ", "world"]
                mock_thread.start.assert_called_once()
                mock_thread.join.assert_called_once_with(timeout=5.0)


class TestTokenCounter:
    """Test TokenCounter class."""

    def test_init_without_tokenizer(self):
        """Test TokenCounter initialization without tokenizer."""
        counter = TokenCounter()
        assert counter.tokenizer is None
        assert counter.total_prompt_tokens == 0
        assert counter.total_completion_tokens == 0

    def test_init_with_tokenizer(self):
        """Test TokenCounter initialization with tokenizer."""
        mock_tokenizer = Mock()
        counter = TokenCounter(mock_tokenizer)
        assert counter.tokenizer == mock_tokenizer

    def test_count_tokens_with_tokenizer(self):
        """Test token counting with tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        counter = TokenCounter(mock_tokenizer)
        count = counter.count_tokens("Hello world")
        
        assert count == 5
        mock_tokenizer.encode.assert_called_once_with("Hello world", add_special_tokens=False)

    def test_count_tokens_without_tokenizer(self):
        """Test token counting without tokenizer (fallback)."""
        counter = TokenCounter()
        count = counter.count_tokens("Hello world test")
        
        # Should use fallback: 3 words * 1.3 = 3.9 -> 3
        assert count == 3

    def test_count_tokens_tokenizer_error(self):
        """Test token counting when tokenizer fails."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        counter = TokenCounter(mock_tokenizer)
        count = counter.count_tokens("Hello world")
        
        # Should fall back to word-based estimation
        assert count == 2  # 2 words * 1.3 = 2.6 -> 2

    def test_count_image_tokens(self):
        """Test counting tokens for images."""
        counter = TokenCounter()
        
        images = [
            {"original_size": (500, 500), "detail_level": "low"},
            {"original_size": (1000, 1000), "detail_level": "high"}
        ]
        
        tokens = counter.count_image_tokens(images)
        
        # Should calculate based on image sizes and detail levels
        assert tokens > 0

    def test_count_image_tokens_fallback(self):
        """Test counting tokens for images without size info."""
        counter = TokenCounter()
        
        images = [
            {"url": "test.jpg"},  # No size info
            {"url": "test2.jpg"}
        ]
        
        tokens = counter.count_image_tokens(images)
        
        # Should use fallback: 2 images * 85 tokens each
        assert tokens == 170

    def test_create_usage(self):
        """Test creating Usage object."""
        counter = TokenCounter()
        
        usage = counter.create_usage(10, 20)
        
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        
        # Check session stats updated
        assert counter.total_prompt_tokens == 10
        assert counter.total_completion_tokens == 20
        assert counter.session_stats["requests_processed"] == 1

    def test_get_session_stats(self):
        """Test getting session statistics."""
        counter = TokenCounter()
        counter.create_usage(10, 20)
        counter.create_usage(5, 15)
        
        stats = counter.get_session_stats()
        
        assert stats["requests_processed"] == 2
        assert stats["total_prompt_tokens"] == 15
        assert stats["total_completion_tokens"] == 35
        assert stats["total_tokens"] == 50


class TestGenerationParameters:
    """Test GenerationParameters class."""

    def test_init_default(self):
        """Test GenerationParameters initialization with defaults."""
        params = GenerationParameters()
        
        assert params.temperature == 0.7
        assert params.max_tokens is None
        assert params.top_p == 1.0
        assert params.top_k == 50
        assert params.repetition_penalty == 1.0
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0
        assert params.stop_sequences == []

    def test_init_custom(self):
        """Test GenerationParameters initialization with custom values."""
        params = GenerationParameters(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            presence_penalty=0.5,
            frequency_penalty=-0.5,
            stop_sequences=["END", "STOP"]
        )
        
        assert params.temperature == 0.5
        assert params.max_tokens == 100
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.repetition_penalty == 1.1
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == -0.5
        assert params.stop_sequences == ["END", "STOP"]

    def test_temperature_validation_invalid(self):
        """Test temperature validation with invalid values."""
        with pytest.raises(ValidationError):
            GenerationParameters(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            GenerationParameters(temperature=2.1)

    def test_top_p_validation_invalid(self):
        """Test top_p validation with invalid values."""
        with pytest.raises(ValidationError):
            GenerationParameters(top_p=0.0)
        
        with pytest.raises(ValidationError):
            GenerationParameters(top_p=1.1)

    def test_parameter_clamping(self):
        """Test parameter clamping for out-of-range values."""
        params = GenerationParameters(
            top_k=0,  # Should be clamped to 1
            repetition_penalty=0.05,  # Should be clamped to 0.1
            presence_penalty=-3.0,  # Should be clamped to -2.0
            frequency_penalty=3.0   # Should be clamped to 2.0
        )
        
        assert params.top_k == 1
        assert params.repetition_penalty == 0.1
        assert params.presence_penalty == -2.0
        assert params.frequency_penalty == 2.0

    def test_to_generation_config(self):
        """Test conversion to GenerationConfig."""
        params = GenerationParameters(
            temperature=0.8,
            max_tokens=50,
            top_p=0.9,
            top_k=40
        )
        
        config = params.to_generation_config(model_max_length=2048)
        
        assert isinstance(config, GenerationConfig)
        assert config.max_new_tokens == 50
        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.do_sample is True  # Because temperature > 0

    def test_to_generation_config_no_sampling(self):
        """Test conversion to GenerationConfig with temperature=0."""
        params = GenerationParameters(temperature=0.0)
        
        config = params.to_generation_config()
        
        assert config.do_sample is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = GenerationParameters(
            temperature=0.5,
            max_tokens=100,
            stop_sequences=["END"]
        )
        
        result = params.to_dict()
        
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 100
        assert result["stop_sequences"] == ["END"]


class TestResponseGenerator:
    """Test ResponseGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.is_loaded.return_value = True
        self.mock_model_manager.model_name = "test-model"
        self.mock_model_manager.tokenizer = Mock()
        self.mock_model_manager.model = Mock()
        
        self.mock_message_processor = Mock(spec=MessageProcessor)
        self.mock_image_handler = Mock(spec=ImageHandler)
        
        self.generator = ResponseGenerator(
            model_manager=self.mock_model_manager,
            message_processor=self.mock_message_processor,
            image_handler=self.mock_image_handler
        )

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        await self.generator.initialize()
        
        # Should initialize token counter with model tokenizer
        assert self.generator.token_counter.tokenizer == self.mock_model_manager.tokenizer

    @pytest.mark.asyncio
    async def test_initialize_model_not_loaded(self):
        """Test initialization when model is not loaded."""
        self.mock_model_manager.is_loaded.return_value = False
        self.mock_model_manager.load_model = AsyncMock()
        
        await self.generator.initialize()
        
        self.mock_model_manager.load_model.assert_called_once()

    def test_normalize_stop_sequences(self):
        """Test normalizing stop sequences."""
        # Test None
        result = self.generator._normalize_stop_sequences(None)
        assert result == []
        
        # Test string
        result = self.generator._normalize_stop_sequences("END")
        assert result == ["END"]
        
        # Test list
        result = self.generator._normalize_stop_sequences(["END", "STOP"])
        assert result == ["END", "STOP"]
        
        # Test list with too many items (should limit to 4)
        result = self.generator._normalize_stop_sequences(["1", "2", "3", "4", "5", "6"])
        assert result == ["1", "2", "3", "4"]

    @pytest.mark.asyncio
    async def test_process_images(self):
        """Test processing images for model input."""
        image_data = [
            {"url": "https://example.com/image.jpg", "detail": "high"},
            {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "detail": "low"}
        ]
        
        # Mock image handler responses
        mock_tensor1 = torch.randn(3, 336, 336)
        mock_tensor2 = torch.randn(3, 336, 336)
        
        self.mock_image_handler.process_image_url = AsyncMock(side_effect=[
            {"tensor": mock_tensor1},
            {"tensor": mock_tensor2}
        ])
        
        result = await self.generator._process_images(image_data)
        
        assert len(result) == 2
        assert torch.equal(result[0], mock_tensor1)
        assert torch.equal(result[1], mock_tensor2)

    @pytest.mark.asyncio
    async def test_process_images_with_error(self):
        """Test processing images when some fail."""
        image_data = [
            {"url": "https://example.com/good.jpg", "detail": "high"},
            {"url": "https://example.com/bad.jpg", "detail": "high"}
        ]
        
        mock_tensor = torch.randn(3, 336, 336)
        
        self.mock_image_handler.process_image_url = AsyncMock(side_effect=[
            {"tensor": mock_tensor},
            Exception("Failed to process image")
        ])
        
        result = await self.generator._process_images(image_data)
        
        # Should only return successful results
        assert len(result) == 1
        assert torch.equal(result[0], mock_tensor)

    def test_prepare_model_inputs(self):
        """Test preparing model inputs."""
        # Mock tokenizer
        self.mock_model_manager.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # Mock model device
        mock_param = Mock()
        mock_param.device = "cpu"
        self.mock_model_manager.model.parameters.return_value = iter([mock_param])
        
        # Test without images
        inputs = self.generator._prepare_model_inputs("Hello world", [])
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "images" not in inputs

    def test_prepare_model_inputs_with_images(self):
        """Test preparing model inputs with images."""
        # Mock tokenizer
        self.mock_model_manager.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # Mock model device
        mock_param = Mock()
        mock_param.device = "cpu"
        self.mock_model_manager.model.parameters.return_value = iter([mock_param])
        
        # Test with images
        image_tensors = [torch.randn(3, 336, 336), torch.randn(3, 336, 336)]
        inputs = self.generator._prepare_model_inputs("Hello world", image_tensors)
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "images" in inputs
        assert inputs["images"].shape == (2, 3, 336, 336)

    def test_apply_stop_sequences(self):
        """Test applying stop sequences to text."""
        text = "Hello world END this should be removed"
        stop_sequences = ["END", "STOP"]
        
        result = self.generator._apply_stop_sequences(text, stop_sequences)
        
        assert result == "Hello world "

    def test_apply_stop_sequences_no_match(self):
        """Test applying stop sequences when no match found."""
        text = "Hello world"
        stop_sequences = ["END", "STOP"]
        
        result = self.generator._apply_stop_sequences(text, stop_sequences)
        
        assert result == "Hello world"

    def test_apply_stop_sequences_empty(self):
        """Test applying empty stop sequences."""
        text = "Hello world"
        stop_sequences = []
        
        result = self.generator._apply_stop_sequences(text, stop_sequences)
        
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_generate_complete_response(self):
        """Test generating complete (non-streaming) response."""
        # Setup mocks
        self.mock_model_manager.ensure_loaded = Mock()
        self.mock_message_processor.process_messages.return_value = {
            "text": "Human: Hello",
            "has_images": False
        }
        self.mock_message_processor.format_for_generation.return_value = "Human: Hello\n\nAssistant: "
        self.mock_message_processor.extract_assistant_response.return_value = "Hi there!"
        
        # Mock model device
        mock_param = Mock()
        mock_param.device = "cpu"
        self.mock_model_manager.model.parameters.return_value = iter([mock_param])
        
        # Mock tokenizer
        self.mock_model_manager.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # Mock token counting
        self.generator.token_counter.count_tokens = Mock(side_effect=[10, 5])  # prompt, completion
        self.generator.token_counter.create_usage = Mock(return_value=Usage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ))
        
        # Mock model generation
        with patch.object(self.generator, '_run_model_generation', return_value="Human: Hello\n\nAssistant: Hi there!"):
            request = ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")]
            )
            
            result = await self.generator._generate_complete_response(
                "Human: Hello\n\nAssistant: ", 
                GenerationParameters(), 
                10, 
                []
            )
            
            assert result["text"] == "Hi there!"
            assert result["finish_reason"] == "stop"
            assert "usage" in result
            assert "generation_time" in result

    def test_get_generation_stats(self):
        """Test getting generation statistics."""
        self.generator.token_counter.get_session_stats = Mock(return_value={
            "requests_processed": 5,
            "total_tokens": 100
        })
        
        stats = self.generator.get_generation_stats()
        
        assert stats["model_loaded"] is True
        assert stats["model_name"] == "test-model"
        assert stats["requests_processed"] == 5
        assert stats["total_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_sse_stream(self):
        """Test SSE streaming response generation."""
        # Setup mocks
        self.mock_model_manager.ensure_loaded = Mock()
        self.mock_message_processor.process_messages.return_value = {
            "text": "Human: Hello",
            "has_images": False
        }
        self.mock_message_processor.format_for_generation.return_value = "Human: Hello\n\nAssistant: "
        
        # Mock streaming response
        async def mock_streaming_response():
            yield {"text": "Hello", "finish_reason": None, "model_name": "test-model"}
            yield {"text": " world", "finish_reason": None, "model_name": "test-model"}
            yield {"text": "", "finish_reason": "stop", "usage": Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7), "model_name": "test-model"}
        
        with patch.object(self.generator, 'generate_response', return_value=mock_streaming_response()):
            request = ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")],
                stream=True
            )
            
            chunks = []
            async for chunk in self.generator.generate_sse_stream(request):
                chunks.append(chunk)
            
            # Should have 3 data chunks + [DONE]
            assert len(chunks) == 4
            assert all(chunk.startswith("data: ") for chunk in chunks[:-1])
            assert chunks[-1] == "data: [DONE]\n\n"
            
            # Check content in chunks
            assert "Hello" in chunks[0]
            assert "world" in chunks[1]
            assert "stop" in chunks[2]

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method."""
        self.mock_image_handler.close = AsyncMock()
        
        await self.generator.cleanup()
        
        self.mock_image_handler.close.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_generation_request_valid(self):
        """Test validating valid generation request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        # Should not raise any exception
        validate_generation_request(request)

    def test_validate_generation_request_empty_messages(self):
        """Test validating request with empty messages."""
        # Since Pydantic already validates this, we test the validation error directly
        with pytest.raises(Exception):  # Pydantic ValidationError
            ChatCompletionRequest(
                model="test-model",
                messages=[]
            )

    def test_validate_generation_request_invalid_temperature(self):
        """Test validating request with invalid temperature."""
        # Since Pydantic already validates this, we test the validation error directly
        with pytest.raises(Exception):  # Pydantic ValidationError
            ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")],
                temperature=2.5
            )

    def test_validate_generation_request_invalid_max_tokens(self):
        """Test validating request with invalid max_tokens."""
        # Since Pydantic already validates this, we test the validation error directly
        with pytest.raises(Exception):  # Pydantic ValidationError
            ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")],
                max_tokens=-10
            )

    def test_validate_generation_request_invalid_top_p(self):
        """Test validating request with invalid top_p."""
        # Since Pydantic already validates this, we test the validation error directly
        with pytest.raises(Exception):  # Pydantic ValidationError
            ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")],
                top_p=1.5
            )

    def test_estimate_generation_time_7b(self):
        """Test estimating generation time for 7B model."""
        time_estimate = estimate_generation_time(
            prompt_tokens=100,
            max_tokens=50,
            model_size="7B"
        )
        
        # Should be prompt overhead + generation time
        expected = (100 * 0.001) + (50 * 0.05)  # 0.1 + 2.5 = 2.6
        assert time_estimate == expected

    def test_estimate_generation_time_13b(self):
        """Test estimating generation time for 13B model."""
        time_estimate = estimate_generation_time(
            prompt_tokens=100,
            max_tokens=50,
            model_size="13B"
        )
        
        # Should be prompt overhead + generation time
        expected = (100 * 0.001) + (50 * 0.08)  # 0.1 + 4.0 = 4.1
        assert time_estimate == expected

    def test_estimate_generation_time_unknown_model(self):
        """Test estimating generation time for unknown model size."""
        time_estimate = estimate_generation_time(
            prompt_tokens=100,
            max_tokens=50,
            model_size="unknown"
        )
        
        # Should use default 7B timing
        expected = (100 * 0.001) + (50 * 0.05)  # 0.1 + 2.5 = 2.6
        assert time_estimate == expected


class TestResponseGeneratorIntegration:
    """Integration tests for ResponseGenerator."""

    @pytest.mark.asyncio
    async def test_full_generation_flow_text_only(self):
        """Test complete generation flow with text-only input."""
        # Create mocks
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.is_loaded.return_value = True
        mock_model_manager.ensure_loaded = Mock()
        mock_model_manager.model_name = "test-model"
        mock_model_manager.tokenizer = Mock()
        mock_model_manager.model = Mock()
        
        # Mock model device
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_model_manager.model.parameters.return_value = iter([mock_param])
        
        # Mock tokenizer
        mock_model_manager.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        mock_model_manager.tokenizer.pad_token_id = 0
        mock_model_manager.tokenizer.eos_token_id = 2
        mock_model_manager.tokenizer.decode.return_value = "Human: Hello\n\nAssistant: Hi there!"
        mock_model_manager.tokenizer.encode.return_value = [1, 2, 3, 4]
        
        # Mock model generation
        mock_output = Mock()
        mock_output.sequences = [torch.tensor([1, 2, 3, 4, 5, 6])]
        mock_model_manager.model.generate.return_value = mock_output
        
        mock_message_processor = Mock(spec=MessageProcessor)
        mock_message_processor.process_messages.return_value = {
            "text": "Human: Hello",
            "has_images": False
        }
        mock_message_processor.format_for_generation.return_value = "Human: Hello\n\nAssistant: "
        mock_message_processor.extract_assistant_response.return_value = "Hi there!"
        
        # Create generator
        generator = ResponseGenerator(
            model_manager=mock_model_manager,
            message_processor=mock_message_processor
        )
        
        await generator.initialize()
        
        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=50
        )
        
        # Generate response
        result = await generator.generate_response(request, stream=False)
        
        # Verify result
        assert result["text"] == "Hi there!"
        assert result["finish_reason"] == "stop"
        assert "usage" in result
        assert "generation_time" in result

    @pytest.mark.asyncio
    async def test_error_handling_model_not_loaded(self):
        """Test error handling when model is not loaded."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.ensure_loaded.side_effect = ModelError("Model not loaded")
        
        generator = ResponseGenerator(model_manager=mock_model_manager)
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(ModelError):
            await generator.generate_response(request)