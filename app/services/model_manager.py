"""Model Manager for Hugging Face A.X-4.0-VL-Light model loading and management."""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from enum import Enum
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    import accelerate
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None
from app.config import settings
from app.schemas.error_models import ModelNotLoadedError, ModelError


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model loading status."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelManager:
    """Manages Hugging Face model loading, health checking, and inference."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize ModelManager.
        
        Args:
            model_name: Name of the Hugging Face model to load
        """
        self.model_name = model_name or settings.model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[AutoProcessor] = None
        
        self.status = ModelStatus.NOT_LOADED
        self.load_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.device = self._determine_device()
        
        # Model configuration
        self.model_config = {
            "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
            "device_map": "auto" if self.device == "auto" else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization if GPU memory is limited and bitsandbytes is available
        if (self.device.startswith("cuda") and 
            settings.gpu_memory_fraction < 1.0 and 
            HAS_BITSANDBYTES and 
            BitsAndBytesConfig is not None):
            try:
                self.model_config["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("4-bit quantization enabled for GPU memory optimization")
            except Exception as e:
                logger.warning(f"Failed to enable quantization: {e}. Proceeding without quantization.")
        
        self._lock = asyncio.Lock()
        
    def _determine_device(self) -> str:
        """Determine the best device for model loading."""
        if settings.device != "auto":
            return settings.device
            
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f}GB)")
            
            if gpu_memory_gb >= 8.0:  # Minimum recommended for VL models
                return "cuda"
            else:
                logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) may be insufficient for optimal performance")
                return "cuda"  # Still try CUDA but with quantization
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"
    
    async def load_model(self) -> None:
        """Load the Hugging Face model, tokenizer, and processor."""
        async with self._lock:
            if self.status == ModelStatus.LOADED:
                logger.info(f"Model {self.model_name} already loaded")
                return
                
            if self.status == ModelStatus.LOADING:
                logger.info(f"Model {self.model_name} is already loading")
                return
                
            self.status = ModelStatus.LOADING
            self.error_message = None
            start_time = time.time()
            
            try:
                logger.info(f"Loading model {self.model_name} on device {self.device}")
                
                # Load tokenizer first (fastest)
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"  # Important for batch inference
                )
                
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load processor for vision-language tasks
                logger.info("Loading processor...")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.warning(f"Could not load processor: {e}. Using tokenizer only.")
                    self.processor = None
                
                # Load model (most memory intensive)
                logger.info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **self.model_config
                )
                
                # Move to specific device if not using device_map
                if self.model_config.get("device_map") is None:
                    self.model = self.model.to(self.device)
                
                # Set to evaluation mode
                self.model.eval()
                
                # Enable memory efficient attention if available
                if hasattr(self.model.config, "use_flash_attention_2"):
                    self.model.config.use_flash_attention_2 = True
                
                self.load_time = time.time() - start_time
                self.status = ModelStatus.LOADED
                
                # Log model info
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                logger.info(f"Model {self.model_name} loaded successfully in {self.load_time:.2f}s")
                logger.info(f"Total parameters: {total_params:,}")
                logger.info(f"Trainable parameters: {trainable_params:,}")
                # Get first parameter device
                try:
                    first_param = next(iter(self.model.parameters()))
                    logger.info(f"Device: {first_param.device}")
                except StopIteration:
                    logger.info("Device: No parameters found")
                
                # Log chat template info
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    logger.info("Chat template available - using tokenizer's chat template")
                else:
                    logger.warning("No chat template found - using fallback format")
                
                # Log special tokens
                logger.info(f"Special tokens - EOS: {self.tokenizer.eos_token}, PAD: {self.tokenizer.pad_token}")
                
                # Log GPU memory usage if using CUDA
                if self.device.startswith("cuda"):
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(f"GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
            except Exception as e:
                self.status = ModelStatus.ERROR
                self.error_message = str(e)
                logger.error(f"Failed to load model {self.model_name}: {e}")
                
                # Clean up partially loaded components
                self.model = None
                self.tokenizer = None
                self.processor = None
                
                raise ModelError(f"Failed to load model {self.model_name}: {e}")
    
    async def unload_model(self) -> None:
        """Unload the model to free memory."""
        async with self._lock:
            if self.status != ModelStatus.LOADED:
                return
                
            logger.info(f"Unloading model {self.model_name}")
            
            # Delete model components
            if self.model is not None:
                del self.model
                self.model = None
                
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear GPU cache if using CUDA
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                
            self.status = ModelStatus.NOT_LOADED
            self.load_time = None
            self.error_message = None
            
            logger.info(f"Model {self.model_name} unloaded successfully")
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.status == ModelStatus.LOADED
    
    def is_loading(self) -> bool:
        """Check if the model is currently loading."""
        return self.status == ModelStatus.LOADING
    
    def has_error(self) -> bool:
        """Check if there was an error loading the model."""
        return self.status == ModelStatus.ERROR
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed model status information."""
        status_info = {
            "model_name": self.model_name,
            "status": self.status.value,
            "device": self.device,
            "load_time": self.load_time,
            "error_message": self.error_message,
        }
        
        if self.is_loaded():
            model_device = None
            if self.model:
                try:
                    first_param = next(iter(self.model.parameters()))
                    model_device = str(first_param.device)
                except StopIteration:
                    model_device = "unknown"
            
            status_info.update({
                "has_tokenizer": self.tokenizer is not None,
                "has_processor": self.processor is not None,
                "model_device": model_device,
            })
            
            # Add GPU memory info if using CUDA
            if self.device.startswith("cuda") and torch.cuda.is_available():
                status_info.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                })
        
        return status_info
    
    def check_health(self) -> Dict[str, Any]:
        """Perform a health check on the model."""
        health_status = {
            "healthy": False,
            "status": self.status.value,
            "checks": {}
        }
        
        # Check if model is loaded
        health_status["checks"]["model_loaded"] = self.is_loaded()
        
        if not self.is_loaded():
            health_status["checks"]["error"] = self.error_message or "Model not loaded"
            return health_status
        
        try:
            # Check tokenizer
            health_status["checks"]["tokenizer_available"] = self.tokenizer is not None
            
            # Check processor (optional for vision tasks)
            health_status["checks"]["processor_available"] = self.processor is not None
            
            # Check model device
            if self.model:
                try:
                    first_param = next(iter(self.model.parameters()))
                    model_device = first_param.device
                    health_status["checks"]["model_on_device"] = str(model_device) == self.device or self.device == "auto"
                    health_status["checks"]["model_device"] = str(model_device)
                except StopIteration:
                    health_status["checks"]["model_on_device"] = False
                    health_status["checks"]["model_device"] = "unknown"
            
            # Check GPU memory if using CUDA
            if self.device.startswith("cuda") and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_usage_percent = (memory_allocated / total_memory) * 100
                
                health_status["checks"]["gpu_memory_usage_percent"] = memory_usage_percent
                health_status["checks"]["gpu_memory_healthy"] = memory_usage_percent < 95.0  # Leave some headroom
            
            # Simple inference test
            if self.tokenizer:
                test_input = "Hello"
                tokens = self.tokenizer.encode(test_input, return_tensors="pt")
                health_status["checks"]["tokenizer_functional"] = tokens.shape[1] > 0
            
            # Overall health check
            health_status["healthy"] = all([
                health_status["checks"]["model_loaded"],
                health_status["checks"]["tokenizer_available"],
                health_status["checks"].get("tokenizer_functional", True),
                health_status["checks"].get("gpu_memory_healthy", True),
            ])
            
        except Exception as e:
            health_status["checks"]["error"] = str(e)
            health_status["healthy"] = False
        
        return health_status
    
    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, raise exception if not."""
        if not self.is_loaded():
            if self.has_error():
                raise ModelError(f"Model failed to load: {self.error_message}")
            else:
                raise ModelNotLoadedError(f"Model {self.model_name} is not loaded")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        if not self.is_loaded():
            raise ModelNotLoadedError("Model must be loaded to get model info")
        
        info = {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type if hasattr(self.model.config, 'model_type') else "unknown",
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', None),
            "hidden_size": getattr(self.model.config, 'hidden_size', None),
            "num_attention_heads": getattr(self.model.config, 'num_attention_heads', None),
            "num_hidden_layers": getattr(self.model.config, 'num_hidden_layers', None),
        }
        
        # Add parameter counts
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            })
        
        return info
    
    def has_chat_template(self) -> bool:
        """Check if the tokenizer has a chat template."""
        if not self.tokenizer:
            return False
        return hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
    
    def prepare_inputs(
        self, 
        messages: List[Dict[str, Any]], 
        images: List[Any] = None
    ) -> Dict[str, Any]:
        """Prepare model inputs using processor's conversations format (official method).
        
        Args:
            messages: List of message dictionaries in OpenAI format
            images: List of images (PIL Images, URLs, or base64 strings)
            
        Returns:
            Processor inputs ready for model generation
        """
        if not self.processor and not self.tokenizer:
            raise ModelNotLoadedError("Processor/Tokenizer not loaded")
        
        # Keep original messages for fallback
        original_messages = messages.copy()
        
        try:
            # Use processor if available (preferred for VL models)
            if self.processor:
                logger.debug(f"Preparing inputs with processor. Messages: {len(messages)}")
                logger.debug(f"First message type: {type(messages[0]) if messages else 'None'}")
                logger.debug(f"First message content: {messages[0] if messages else 'None'}")
                
                # Check if messages are already in A.X-4.0-VL-Light format
                # (i.e., content is already a list with type/text structure)
                if (messages and isinstance(messages[0].get('content'), list) and 
                    messages[0]['content'] and 'type' in messages[0]['content'][0]):
                    logger.debug("Messages already in A.X-4.0-VL-Light format, using directly")
                    ax_messages = messages
                else:
                    # Convert to official A.X-4.0-VL-Light format
                    try:
                        ax_messages = self._convert_to_ax_format(messages)
                        logger.debug(f"Converted messages: {ax_messages}")
                    except Exception as e:
                        logger.error(f"Failed in _convert_to_ax_format: {e}")
                        raise e
                    
                conversations = [ax_messages]
                
                # Process images if provided
                processed_images = []
                if images:
                    logger.info(f"Processing {len(images)} images for A.X-4.0-VL-Light")
                    try:
                        processed_images = self._process_images_for_processor(images)
                    except Exception as e:
                        logger.error(f"Failed in _process_images_for_processor: {e}")
                        raise e
                
                logger.debug(f"About to call processor with conversations: {conversations}")
                try:
                    inputs = self.processor(
                        images=processed_images,
                        conversations=conversations,
                        padding=True,
                        return_tensors="pt"
                    )
                except Exception as e:
                    logger.error(f"Failed in processor call: {e}")
                    logger.error(f"Conversations: {conversations}")
                    logger.error(f"Images count: {len(processed_images)}")
                    raise e
                logger.debug(f"Using processor with conversations format. Images: {len(processed_images)}")
                return inputs
            
            # Fallback to tokenizer chat template
            elif self.tokenizer and self.has_chat_template():
                # Use original OpenAI format for chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    original_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True
                )
                logger.debug("Using tokenizer chat template")
                return inputs
                
            else:
                # Final fallback - convert to simple format
                simple_messages = []
                for msg in original_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Extract text from multi-modal content
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = " ".join(text_parts)
                    simple_messages.append({"role": role, "content": content})
                
                formatted_text = self._fallback_chat_format(simple_messages, True)
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True
                )
                logger.debug("Using fallback chat format")
                return inputs
                
        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise ModelError(f"Input preparation failed: {str(e)}")
    
    def _convert_to_ax_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI format messages to A.X-4.0-VL-Light format.
        
        Args:
            messages: OpenAI format messages
            
        Returns:
            A.X-4.0-VL-Light format messages
        """
        ax_messages = []
        
        for msg in messages:
            try:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if isinstance(content, str):
                    # Simple text message
                    ax_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Multi-modal content (already in correct format or needs validation)
                    # Ensure all items in content list are valid
                    validated_content = []
                    for item in content:
                        if isinstance(item, dict) and "type" in item:
                            validated_content.append(item)
                        else:
                            logger.warning(f"Invalid content item: {item}, skipping")
                    
                    ax_messages.append({
                        "role": role,
                        "content": validated_content
                    })
                else:
                    logger.warning(f"Unknown content type: {type(content)} for role {role}")
                    # Default to empty text
                    ax_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": ""}]
                    })
            except Exception as e:
                logger.error(f"Error processing message {msg}: {e}")
                # Skip this message or add default
                ax_messages.append({
                    "role": msg.get("role", "user"),
                    "content": [{"type": "text", "text": ""}]
                })
        
        return ax_messages
    
    def _fallback_chat_format(
        self, 
        messages: List[Dict[str, str]], 
        add_generation_prompt: bool = True
    ) -> str:
        """Fallback chat formatting for models without chat template."""
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_messages.append(f"<|system|>\n{content}")
            elif role == "user":
                formatted_messages.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted_messages.append(f"<|assistant|>\n{content}")
        
        result = "\n".join(formatted_messages)
        
        if add_generation_prompt:
            result += "\n<|assistant|>\n"
        
        return result
    
    def _process_images_for_processor(self, image_urls: List[str]) -> List[Any]:
        """Process image URLs to format suitable for A.X-4.0-VL-Light processor.
        
        Args:
            image_urls: List of image URLs (base64 data URLs or HTTP URLs)
            
        Returns:
            List of processed images (PIL Images)
        """
        import base64
        import io
        from PIL import Image
        import requests
        
        processed_images = []
        
        for url in image_urls:
            try:
                if url.startswith('data:image/'):
                    # Handle base64 data URL
                    logger.debug(f"Processing base64 image URL: {url[:100]}...")
                    
                    # Extract base64 data
                    header, encoded = url.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    
                    # Create PIL Image
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    processed_images.append(image)
                    logger.info(f"Successfully processed base64 image: {image.size}")
                    
                elif url.startswith('http'):
                    # Handle HTTP URL
                    logger.debug(f"Processing HTTP image URL: {url}")
                    
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    image = Image.open(io.BytesIO(response.content))
                    
                    # Convert to RGB if necessary  
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    processed_images.append(image)
                    logger.info(f"Successfully processed HTTP image: {image.size}")
                    
                else:
                    logger.warning(f"Unsupported image URL format: {url[:100]}...")
                    
            except Exception as e:
                logger.error(f"Failed to process image URL {url[:100]}...: {e}")
                continue
        
        logger.info(f"Processed {len(processed_images)} images successfully")
        return processed_images


# Global model manager instance
model_manager = ModelManager()