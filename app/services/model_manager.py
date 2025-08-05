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
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
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
            HAS_BITSANDBYTES):
            self.model_config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
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


# Global model manager instance
model_manager = ModelManager()