"""Response generation engine for model inference and token management."""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import torch
from transformers import GenerationConfig, TextIteratorStreamer
from threading import Thread

from app.services.model_manager import ModelManager
from app.models.message_processor import MessageProcessor
from app.models.image_handler import ImageHandler
from app.schemas.chat_models import ChatCompletionRequest, Usage
from app.schemas.error_models import ModelError, ValidationError


logger = logging.getLogger(__name__)


class SSEFormatter:
    """Formats responses for Server-Sent Events (SSE) streaming."""
    
    @staticmethod
    def format_chunk(data: Dict[str, Any]) -> str:
        """Format a data chunk for SSE.
        
        Args:
            data: Data to format
            
        Returns:
            SSE formatted string
        """
        json_data = json.dumps(data, ensure_ascii=False)
        return f"data: {json_data}\n\n"
    
    @staticmethod
    def format_done() -> str:
        """Format the final [DONE] message for SSE.
        
        Returns:
            SSE formatted done message
        """
        return "data: [DONE]\n\n"
    
    @staticmethod
    def format_error(error: str) -> str:
        """Format an error message for SSE.
        
        Args:
            error: Error message
            
        Returns:
            SSE formatted error message
        """
        error_data = {
            "error": {
                "message": error,
                "type": "server_error"
            }
        }
        return SSEFormatter.format_chunk(error_data)


class StreamingTokenizer:
    """Handles token-by-token streaming generation."""
    
    def __init__(self, tokenizer, model):
        """Initialize StreamingTokenizer.
        
        Args:
            tokenizer: Hugging Face tokenizer
            model: Hugging Face model
        """
        self.tokenizer = tokenizer
        self.model = model
        self.streamer = None
    
    def create_streamer(self, skip_prompt: bool = True) -> TextIteratorStreamer:
        """Create a TextIteratorStreamer for token streaming.
        
        Args:
            skip_prompt: Whether to skip prompt tokens in output
            
        Returns:
            TextIteratorStreamer instance
        """
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
            timeout=30.0
        )
        return self.streamer
    
    async def generate_streaming(
        self,
        inputs: Dict[str, Any],
        generation_config: GenerationConfig
    ) -> AsyncGenerator[str, None]:
        """Generate streaming tokens.
        
        Args:
            inputs: Model inputs
            generation_config: Generation configuration
            
        Yields:
            Generated tokens
        """
        streamer = self.create_streamer()
        
        # Run generation in a separate thread
        generation_kwargs = {
            **inputs,
            "generation_config": generation_config,
            "streamer": streamer,
            "return_dict_in_generate": True,
            "output_scores": False
        }
        
        def _generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)
        
        # Start generation thread
        thread = Thread(target=_generate)
        thread.start()
        
        try:
            # Stream tokens as they are generated
            for token in streamer:
                if token:  # Skip empty tokens
                    yield token
                await asyncio.sleep(0.001)  # Small delay to prevent blocking
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
        finally:
            # Ensure thread completes
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Generation thread did not complete in time")


class TokenCounter:
    """Handles token counting and usage tracking."""
    
    def __init__(self, tokenizer=None):
        """Initialize TokenCounter.
        
        Args:
            tokenizer: Hugging Face tokenizer for accurate counting
        """
        self.tokenizer = tokenizer
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.session_stats = {
            "requests_processed": 0,
            "total_tokens": 0,
            "average_response_time": 0.0
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, using fallback: {e}")
        
        # Fallback estimation
        return max(1, int(len(text.split()) * 1.3))
    
    def count_image_tokens(self, images: List[Dict[str, Any]]) -> int:
        """Count tokens for image processing.
        
        Args:
            images: List of processed image data
            
        Returns:
            Total image tokens
        """
        total_tokens = 0
        for image_data in images:
            if "original_size" in image_data:
                width, height = image_data["original_size"]
                detail = image_data.get("detail_level", "auto")
                
                # Use the same estimation as ImageHandler
                from app.models.image_handler import estimate_image_tokens
                tokens = estimate_image_tokens(width, height, detail)
                total_tokens += tokens
            else:
                # Fallback for unknown image
                total_tokens += 85  # Base image tokens
        
        return total_tokens
    
    def create_usage(
        self, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> Usage:
        """Create Usage object for response.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Usage object
        """
        total_tokens = prompt_tokens + completion_tokens
        
        # Update session stats
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.session_stats["requests_processed"] += 1
        self.session_stats["total_tokens"] += total_tokens
        
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        return {
            **self.session_stats,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


class GenerationParameters:
    """Manages and validates generation parameters."""
    
    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None
    ):
        """Initialize generation parameters.
        
        Args:
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty (1.0 = no penalty)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            stop_sequences: List of stop sequences
        """
        self.temperature = self._validate_temperature(temperature)
        self.max_tokens = max_tokens
        self.top_p = self._validate_top_p(top_p)
        self.top_k = max(1, top_k)
        self.repetition_penalty = max(0.1, repetition_penalty)
        self.presence_penalty = max(-2.0, min(2.0, presence_penalty))
        self.frequency_penalty = max(-2.0, min(2.0, frequency_penalty))
        self.do_sample = do_sample
        self.stop_sequences = stop_sequences or []
    
    def _validate_temperature(self, temperature: float) -> float:
        """Validate temperature parameter."""
        if temperature < 0.0 or temperature > 2.0:
            raise ValidationError(
                "Temperature must be between 0.0 and 2.0",
                param="temperature"
            )
        return temperature
    
    def _validate_top_p(self, top_p: float) -> float:
        """Validate top_p parameter."""
        if top_p <= 0.0 or top_p > 1.0:
            raise ValidationError(
                "top_p must be between 0.0 and 1.0",
                param="top_p"
            )
        return top_p
    
    def to_generation_config(self, model_max_length: int = 2048) -> GenerationConfig:
        """Convert to Hugging Face GenerationConfig.
        
        Args:
            model_max_length: Maximum sequence length for the model
            
        Returns:
            GenerationConfig object
        """
        max_new_tokens = self.max_tokens or (model_max_length // 2)
        
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            pad_token_id=None,  # Will be set by the model
            eos_token_id=None,  # Will be set by the model
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_sequences": self.stop_sequences,
        }


class ResponseGenerator:
    """Generates responses using the loaded model with configurable parameters."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        message_processor: Optional[MessageProcessor] = None,
        image_handler: Optional[ImageHandler] = None
    ):
        """Initialize ResponseGenerator.
        
        Args:
            model_manager: ModelManager instance
            message_processor: MessageProcessor for text processing
            image_handler: ImageHandler for image processing
        """
        self.model_manager = model_manager
        self.message_processor = message_processor or MessageProcessor(model_manager)
        self.image_handler = image_handler or ImageHandler()
        self.token_counter = TokenCounter()
        
        # Generation state
        self._generation_lock = asyncio.Lock()
        self._current_generation = None
        self._streaming_tokenizer = None
        
    async def initialize(self):
        """Initialize the response generator."""
        # Ensure model is loaded
        if not self.model_manager.is_loaded():
            await self.model_manager.load_model()
        
        # Initialize token counter with model tokenizer
        self.token_counter = TokenCounter(self.model_manager.tokenizer)
        
        # Initialize streaming tokenizer
        self._streaming_tokenizer = StreamingTokenizer(
            self.model_manager.tokenizer,
            self.model_manager.model
        )
        
        logger.info("ResponseGenerator initialized successfully")
    
    async def generate_response(
        self,
        request: ChatCompletionRequest,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate response for chat completion request.
        
        Args:
            request: ChatCompletionRequest object
            stream: Whether to stream the response
            
        Returns:
            Response dictionary or async generator for streaming
            
        Raises:
            ModelError: If generation fails
            ValidationError: If request is invalid
        """
        try:
            # Ensure model is ready
            self.model_manager.ensure_loaded()
            
            # Process messages using new official format
            processed_input = self.message_processor.process_messages(
                request.messages,
                include_history=True
            )
            
            # Get model inputs (ready for generation)
            model_inputs = self.message_processor.format_for_generation(processed_input)
            
            # Create generation parameters using official recommendations
            gen_params = self._create_official_generation_params(request)
            
            # Count prompt tokens
            prompt_tokens = self._count_input_tokens(model_inputs, processed_input)
            
            if stream:
                return self._generate_streaming_response(
                    model_inputs, gen_params, prompt_tokens, processed_input
                )
            else:
                return await self._generate_complete_response(
                    model_inputs, gen_params, prompt_tokens, processed_input
                )
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise ModelError(f"Failed to generate response: {str(e)}")
    
    def _create_official_generation_params(self, request: ChatCompletionRequest) -> GenerationParameters:
        """Create generation parameters using official A.X-4.0-VL-Light recommendations.
        
        Args:
            request: Chat completion request
            
        Returns:
            GenerationParameters with official settings
        """
        # Official recommended parameters from Hugging Face documentation
        return GenerationParameters(
            temperature=request.temperature if request.temperature is not None else 0.5,  # Official: 0.5
            max_tokens=request.max_tokens or 256,  # Official: 256 for basic responses
            top_p=request.top_p if request.top_p is not None else 0.8,  # Official: 0.8
            top_k=20,  # Official: 20
            repetition_penalty=1.05,  # Official: 1.05
            do_sample=True,  # Official: True
            stop_sequences=self._normalize_stop_sequences(request.stop)
        )
    
    def _count_input_tokens(self, model_inputs: Dict[str, Any], processed_input: Dict[str, Any]) -> int:
        """Count tokens in model inputs.
        
        Args:
            model_inputs: Prepared model inputs
            processed_input: Original processed input
            
        Returns:
            Token count
        """
        try:
            # If we have tensor inputs with input_ids, count those
            if isinstance(model_inputs, dict) and "input_ids" in model_inputs:
                return model_inputs["input_ids"].shape[-1]
            
            # Fallback: try to get text and count
            if "formatted_text" in model_inputs:
                return self.token_counter.count_tokens(model_inputs["formatted_text"])
            
            # Another fallback: use original text
            if "text" in processed_input:
                return self.token_counter.count_tokens(processed_input["text"])
            
            # Default fallback
            logger.warning("Could not determine token count, using default")
            return 50
            
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return 50
    
    async def _process_images(self, image_data_list: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Process images for model input.
        
        Args:
            image_data_list: List of image data dictionaries
            
        Returns:
            List of processed image tensors
        """
        image_tensors = []
        
        for image_data in image_data_list:
            try:
                url = image_data.get("url")
                detail = image_data.get("detail", "auto")
                
                if url:
                    processed = await self.image_handler.process_image_url(url, detail)
                    image_tensors.append(processed["tensor"])
                    
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                # Continue with other images
                continue
        
        return image_tensors
    
    def _normalize_stop_sequences(self, stop: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize stop sequences to list format.
        
        Args:
            stop: Stop sequences from request
            
        Returns:
            List of stop sequences
        """
        if stop is None:
            return []
        elif isinstance(stop, str):
            return [stop]
        elif isinstance(stop, list):
            return stop[:4]  # Limit to 4 stop sequences
        else:
            return []
    
    async def _generate_complete_response(
        self,
        model_inputs: Dict[str, Any],
        gen_params: GenerationParameters,
        prompt_tokens: int,
        processed_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate complete (non-streaming) response using official A.X-4.0-VL-Light format.
        
        Args:
            model_inputs: Prepared model inputs from processor
            gen_params: Generation parameters
            prompt_tokens: Number of prompt tokens
            processed_input: Original processed input for fallback
            
        Returns:
            Complete response dictionary
        """
        async with self._generation_lock:
            start_time = time.time()
            
            try:
                # Use official A.X-4.0-VL-Light generation method
                generated_text = await self._run_official_generation(
                    model_inputs, gen_params
                )
                
                # Extract assistant response with improved logic
                original_prompt = self._extract_original_prompt(model_inputs, processed_input)
                response_text = self.message_processor.extract_assistant_response(
                    generated_text, original_prompt
                )
                
                # Apply stop sequences
                response_text = self._apply_stop_sequences(
                    response_text, gen_params.stop_sequences
                )
                
                # Count completion tokens
                completion_tokens = self.token_counter.count_tokens(response_text)
                
                # Create usage info
                usage = self.token_counter.create_usage(prompt_tokens, completion_tokens)
                
                generation_time = time.time() - start_time
                logger.info(f"Generated response in {generation_time:.2f}s using official A.X-4.0-VL-Light method")
                
                return {
                    "text": response_text,
                    "usage": usage,
                    "generation_time": generation_time,
                    "finish_reason": "stop",
                    "model_name": self.model_manager.model_name
                }
                
            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                raise ModelError(f"Model generation failed: {str(e)}")
    
    async def _run_official_generation(
        self,
        model_inputs: Dict[str, Any],
        gen_params: GenerationParameters
    ) -> str:
        """Run model generation using official A.X-4.0-VL-Light method.
        
        Args:
            model_inputs: Prepared model inputs from processor
            gen_params: Generation parameters
            
        Returns:
            Generated text
        """
        model = self.model_manager.model
        processor = self.model_manager.processor
        tokenizer = self.model_manager.tokenizer
        
        # Move inputs to device
        device = next(model.parameters()).device
        for key, value in model_inputs.items():
            if hasattr(value, 'to'):
                model_inputs[key] = value.to(device)
        
        # Create generation config using official parameters
        generation_config = gen_params.to_generation_config()
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                return outputs.sequences[0]
        
        # Run generation
        output_ids = await loop.run_in_executor(None, _generate)
        
        # Decode output using processor if available, otherwise tokenizer
        if processor:
            # Extract generated tokens (remove input tokens)
            input_length = model_inputs.get('input_ids', torch.tensor([])).shape[-1]
            generated_ids_trimmed = output_ids[input_length:]
            
            generated_text = processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        else:
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _extract_original_prompt(
        self,
        model_inputs: Dict[str, Any],
        processed_input: Dict[str, Any]
    ) -> str:
        """Extract original prompt for response extraction.
        
        Args:
            model_inputs: Prepared model inputs
            processed_input: Original processed input
            
        Returns:
            Original prompt text
        """
        # Try to get from tokenizer decoding
        if "input_ids" in model_inputs:
            try:
                return self.model_manager.tokenizer.decode(
                    model_inputs["input_ids"][0], 
                    skip_special_tokens=True
                )
            except Exception:
                pass
        
        # Fallback to processed input
        if "text" in processed_input:
            return processed_input["text"]
        
        # Last resort
        return ""
    
    async def _generate_streaming_response(
        self,
        model_inputs: Dict[str, Any],
        gen_params: GenerationParameters,
        prompt_tokens: int,
        processed_input: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using official A.X-4.0-VL-Light format.
        
        Args:
            model_inputs: Prepared model inputs from processor
            gen_params: Generation parameters
            prompt_tokens: Number of prompt tokens
            processed_input: Original processed input for fallback
            
        Yields:
            Response chunks
        """
        # For now, fallback to complete generation and stream it word by word
        try:
            complete_response = await self._generate_complete_response(
                model_inputs, gen_params, prompt_tokens, processed_input
            )
            
            response_text = complete_response["text"]
            words = response_text.split()
            
            for i, word in enumerate(words):
                # Add space before word (except first)
                chunk_text = f" {word}" if i > 0 else word
                
                yield {
                    "text": chunk_text,
                    "finish_reason": None,
                    "model_name": self.model_manager.model_name
                }
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
            
            # Final chunk with usage info
            yield {
                "text": "",
                "finish_reason": "stop",
                "usage": complete_response["usage"],
                "model_name": self.model_manager.model_name
            }
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {
                "error": str(e),
                "finish_reason": "error",
                "model_name": self.model_manager.model_name
            }
    
    def _prepare_model_inputs(
        self, 
        prompt: str, 
        image_tensors: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Prepare inputs for model generation.
        
        Args:
            prompt: Text prompt
            image_tensors: List of image tensors
            
        Returns:
            Model input dictionary
        """
        # Tokenize text
        tokenizer = self.model_manager.tokenizer
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to model device
        try:
            device = next(iter(self.model_manager.model.parameters())).device
        except (StopIteration, AttributeError):
            # Fallback to CPU if model parameters are not available
            device = "cpu"
        for key in text_inputs:
            text_inputs[key] = text_inputs[key].to(device)
        
        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"]
        }
        
        # Add images if present
        if image_tensors:
            # Stack image tensors
            images = torch.stack(image_tensors).to(device)
            inputs["images"] = images
        
        return inputs
    
    async def _run_model_generation(
        self,
        inputs: Dict[str, Any],
        gen_params: GenerationParameters
    ) -> str:
        """Run model generation.
        
        Args:
            inputs: Model inputs
            gen_params: Generation parameters
            
        Returns:
            Generated text
        """
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer
        
        # Create generation config
        generation_config = gen_params.to_generation_config()
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                return outputs.sequences[0]
        
        # Run generation
        output_ids = await loop.run_in_executor(None, _generate)
        
        # Decode output
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return generated_text
    
    async def _run_streaming_generation(
        self,
        inputs: Dict[str, Any],
        gen_params: GenerationParameters
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run streaming model generation.
        
        Args:
            inputs: Model inputs
            gen_params: Generation parameters
            
        Yields:
            Generation chunks
        """
        if not self._streaming_tokenizer:
            # Fallback to word-by-word streaming if no streaming tokenizer
            async for chunk in self._fallback_streaming_generation(inputs, gen_params):
                yield chunk
            return
        
        try:
            # Create generation config
            generation_config = gen_params.to_generation_config()
            generation_config.pad_token_id = self.model_manager.tokenizer.pad_token_id
            generation_config.eos_token_id = self.model_manager.tokenizer.eos_token_id
            
            # Stream tokens as they are generated
            async for token in self._streaming_tokenizer.generate_streaming(inputs, generation_config):
                if token:  # Skip empty tokens
                    # Apply stop sequences to accumulated text
                    if gen_params.stop_sequences:
                        # For streaming, we need to check if any stop sequence is encountered
                        should_stop = False
                        for stop_seq in gen_params.stop_sequences:
                            if stop_seq in token:
                                # Split at stop sequence and yield only the part before it
                                parts = token.split(stop_seq, 1)
                                if parts[0]:
                                    yield {"text": parts[0]}
                                should_stop = True
                                break
                        
                        if should_stop:
                            break
                    
                    yield {"text": token}
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Fallback to word-by-word streaming
            async for chunk in self._fallback_streaming_generation(inputs, gen_params):
                yield chunk
    
    async def _fallback_streaming_generation(
        self,
        inputs: Dict[str, Any],
        gen_params: GenerationParameters
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fallback streaming by chunking complete response.
        
        Args:
            inputs: Model inputs
            gen_params: Generation parameters
            
        Yields:
            Generation chunks
        """
        # Generate complete response first
        complete_response = await self._run_model_generation(inputs, gen_params)
        
        # Extract the new part (after the prompt)
        prompt_text = self.model_manager.tokenizer.decode(
            inputs["input_ids"][0], 
            skip_special_tokens=True
        )
        
        if complete_response.startswith(prompt_text):
            new_text = complete_response[len(prompt_text):]
        else:
            new_text = complete_response
        
        # Apply stop sequences
        new_text = self._apply_stop_sequences(new_text, gen_params.stop_sequences)
        
        # Stream word by word
        words = new_text.split()
        for i, word in enumerate(words):
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield {"text": chunk_text}
            await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    def _apply_stop_sequences(self, text: str, stop_sequences: List[str]) -> str:
        """Apply stop sequences to generated text.
        
        Args:
            text: Generated text
            stop_sequences: List of stop sequences
            
        Returns:
            Text with stop sequences applied
        """
        if not stop_sequences:
            return text
        
        # Find the earliest stop sequence
        min_index = len(text)
        for stop_seq in stop_sequences:
            index = text.find(stop_seq)
            if index != -1 and index < min_index:
                min_index = index
        
        if min_index < len(text):
            return text[:min_index]
        
        return text
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            "model_loaded": self.model_manager.is_loaded(),
            "model_name": self.model_manager.model_name,
            **self.token_counter.get_session_stats()
        }
    
    async def generate_sse_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate SSE formatted streaming response.
        
        Args:
            request: ChatCompletionRequest object
            
        Yields:
            SSE formatted response chunks
        """
        try:
            # Generate unique ID for this completion
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())
            
            # Process the request and get streaming generator
            response_generator = await self.generate_response(request, stream=True)
            
            chunk_index = 0
            accumulated_text = ""
            
            async for chunk in response_generator:
                if "error" in chunk:
                    # Send error in SSE format
                    yield SSEFormatter.format_error(chunk["error"])
                    return
                
                chunk_text = chunk.get("text", "")
                finish_reason = chunk.get("finish_reason")
                
                if chunk_text or finish_reason:
                    # Create streaming response chunk
                    sse_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": chunk_text
                            } if chunk_text else {},
                            "finish_reason": finish_reason
                        }]
                    }
                    
                    # Add usage info for final chunk
                    if finish_reason and "usage" in chunk:
                        sse_chunk["usage"] = chunk["usage"].model_dump()
                    
                    yield SSEFormatter.format_chunk(sse_chunk)
                    
                    if chunk_text:
                        accumulated_text += chunk_text
                        chunk_index += 1
                    
                    if finish_reason:
                        break
            
            # Send final [DONE] message
            yield SSEFormatter.format_done()
            
        except Exception as e:
            logger.error(f"SSE streaming failed: {e}")
            yield SSEFormatter.format_error(str(e))
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.image_handler:
            await self.image_handler.close()
        
        logger.info("ResponseGenerator cleaned up")


# Utility functions
def validate_generation_request(request: ChatCompletionRequest) -> None:
    """Validate generation request parameters.
    
    Args:
        request: ChatCompletionRequest to validate
        
    Raises:
        ValidationError: If request is invalid
    """
    if not request.messages:
        raise ValidationError("Messages cannot be empty")
    
    if request.temperature is not None:
        if request.temperature < 0.0 or request.temperature > 2.0:
            raise ValidationError(
                "Temperature must be between 0.0 and 2.0",
                param="temperature"
            )
    
    if request.max_tokens is not None:
        if request.max_tokens <= 0:
            raise ValidationError(
                "max_tokens must be positive",
                param="max_tokens"
            )
    
    if request.top_p is not None:
        if request.top_p <= 0.0 or request.top_p > 1.0:
            raise ValidationError(
                "top_p must be between 0.0 and 1.0",
                param="top_p"
            )


def estimate_generation_time(
    prompt_tokens: int,
    max_tokens: int,
    model_size: str = "7B"
) -> float:
    """Estimate generation time based on token count and model size.
    
    Args:
        prompt_tokens: Number of prompt tokens
        max_tokens: Maximum tokens to generate
        model_size: Model size (7B, 13B, etc.)
        
    Returns:
        Estimated generation time in seconds
    """
    # Base time per token (rough estimates)
    base_times = {
        "7B": 0.05,   # 50ms per token
        "13B": 0.08,  # 80ms per token
        "30B": 0.15,  # 150ms per token
    }
    
    time_per_token = base_times.get(model_size, 0.05)
    
    # Add overhead for prompt processing
    prompt_overhead = prompt_tokens * 0.001  # 1ms per prompt token
    
    # Estimate total time
    total_time = prompt_overhead + (max_tokens * time_per_token)
    
    return total_time