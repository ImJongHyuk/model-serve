"""Unit tests for ModelManager class."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import torch
from app.services.model_manager import ModelManager, ModelStatus
from app.schemas.error_models import ModelNotLoadedError, ModelError


class TestModelManager:
    """Test ModelManager class."""

    def test_init_default_model(self):
        """Test ModelManager initialization with default model."""
        manager = ModelManager()
        assert manager.model_name == "skt/A.X-4.0-VL-Light"  # From settings
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.processor is None
        assert manager.status == ModelStatus.NOT_LOADED
        assert manager.load_time is None
        assert manager.error_message is None

    def test_init_custom_model(self):
        """Test ModelManager initialization with custom model."""
        custom_model = "test/custom-model"
        manager = ModelManager(model_name=custom_model)
        assert manager.model_name == custom_model

    def test_determine_device_auto_with_cuda(self):
        """Test device determination with CUDA available."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.get_device_name', return_value="Test GPU"), \
             patch('app.services.model_manager.HAS_BITSANDBYTES', False):
            
            # Mock GPU with sufficient memory
            mock_props.return_value.total_memory = 16 * (1024**3)  # 16GB
            
            manager = ModelManager()
            assert manager.device == "cuda"

    def test_determine_device_auto_without_cuda(self):
        """Test device determination without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            manager = ModelManager()
            assert manager.device == "cpu"

    def test_determine_device_explicit(self):
        """Test explicit device setting."""
        with patch('app.services.model_manager.settings') as mock_settings:
            mock_settings.device = "cpu"
            mock_settings.model_name = "test/model"
            mock_settings.gpu_memory_fraction = 0.9
            
            manager = ModelManager()
            assert manager.device == "cpu"

    def test_is_loaded_false(self):
        """Test is_loaded when model is not loaded."""
        manager = ModelManager()
        assert not manager.is_loaded()

    def test_is_loading_false(self):
        """Test is_loading when model is not loading."""
        manager = ModelManager()
        assert not manager.is_loading()

    def test_has_error_false(self):
        """Test has_error when there's no error."""
        manager = ModelManager()
        assert not manager.has_error()

    def test_has_error_true(self):
        """Test has_error when there's an error."""
        manager = ModelManager()
        manager.status = ModelStatus.ERROR
        manager.error_message = "Test error"
        assert manager.has_error()

    def test_get_status_not_loaded(self):
        """Test get_status when model is not loaded."""
        manager = ModelManager()
        status = manager.get_status()
        
        expected_keys = ["model_name", "status", "device", "load_time", "error_message"]
        for key in expected_keys:
            assert key in status
        
        assert status["status"] == "not_loaded"
        assert status["model_name"] == manager.model_name

    def test_ensure_loaded_not_loaded(self):
        """Test ensure_loaded raises exception when model not loaded."""
        manager = ModelManager()
        
        with pytest.raises(ModelNotLoadedError):
            manager.ensure_loaded()

    def test_ensure_loaded_error_state(self):
        """Test ensure_loaded raises exception when model has error."""
        manager = ModelManager()
        manager.status = ModelStatus.ERROR
        manager.error_message = "Test error"
        
        with pytest.raises(ModelError):
            manager.ensure_loaded()

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self):
        """Test load_model when model is already loaded."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADED
        
        # Should return early without attempting to load
        await manager.load_model()
        assert manager.status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_load_model_already_loading(self):
        """Test load_model when model is already loading."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADING
        
        # Should return early without attempting to load
        await manager.load_model()
        assert manager.status == ModelStatus.LOADING

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading."""
        manager = ModelManager()
        
        # Mock all the components
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        mock_processor = Mock()
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config = Mock()
        
        # Mock parameters properly
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        with patch('app.services.model_manager.AutoTokenizer') as mock_tokenizer_class, \
             patch('app.services.model_manager.AutoProcessor') as mock_processor_class, \
             patch('app.services.model_manager.AutoModelForCausalLM') as mock_model_class, \
             patch('app.services.model_manager.logger') as mock_logger:
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_model_class.from_pretrained.return_value = mock_model
            
            await manager.load_model()
            
            assert manager.status == ModelStatus.LOADED
            assert manager.tokenizer == mock_tokenizer
            assert manager.processor == mock_processor
            assert manager.model == mock_model
            assert manager.load_time is not None
            assert manager.load_time > 0
            
            # Verify pad token was set
            assert mock_tokenizer.pad_token == "<eos>"

    @pytest.mark.asyncio
    async def test_load_model_processor_failure(self):
        """Test model loading when processor fails to load."""
        manager = ModelManager()
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config = Mock()
        
        # Mock parameters properly
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        with patch('app.services.model_manager.AutoTokenizer') as mock_tokenizer_class, \
             patch('app.services.model_manager.AutoProcessor') as mock_processor_class, \
             patch('app.services.model_manager.AutoModelForCausalLM') as mock_model_class, \
             patch('app.services.model_manager.logger') as mock_logger:
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_processor_class.from_pretrained.side_effect = Exception("Processor not found")
            mock_model_class.from_pretrained.return_value = mock_model
            
            await manager.load_model()
            
            assert manager.status == ModelStatus.LOADED
            assert manager.tokenizer == mock_tokenizer
            assert manager.processor is None  # Should be None due to failure
            assert manager.model == mock_model

    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        """Test model loading failure."""
        manager = ModelManager()
        
        with patch('app.services.model_manager.AutoTokenizer') as mock_tokenizer_class, \
             patch('app.services.model_manager.logger') as mock_logger:
            
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
            
            with pytest.raises(ModelError):
                await manager.load_model()
            
            assert manager.status == ModelStatus.ERROR
            assert manager.error_message == "Model not found"
            assert manager.model is None
            assert manager.tokenizer is None
            assert manager.processor is None

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        """Test unload_model when model is not loaded."""
        manager = ModelManager()
        
        # Should return early without error
        await manager.unload_model()
        assert manager.status == ModelStatus.NOT_LOADED

    @pytest.mark.asyncio
    async def test_unload_model_success(self):
        """Test successful model unloading."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADED
        manager.model = Mock()
        manager.tokenizer = Mock()
        manager.processor = Mock()
        manager.load_time = 1.0
        manager.error_message = None
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            await manager.unload_model()
            
            assert manager.status == ModelStatus.NOT_LOADED
            assert manager.model is None
            assert manager.tokenizer is None
            assert manager.processor is None
            assert manager.load_time is None
            assert manager.error_message is None

    def test_check_health_not_loaded(self):
        """Test health check when model is not loaded."""
        manager = ModelManager()
        
        health = manager.check_health()
        
        assert not health["healthy"]
        assert health["status"] == "not_loaded"
        assert not health["checks"]["model_loaded"]
        assert "error" in health["checks"]

    def test_check_health_loaded_success(self):
        """Test health check when model is loaded successfully."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADED
        
        # Mock components
        mock_tokenizer = Mock()
        mock_tensor = Mock()
        mock_tensor.shape = (1, 5)  # Mock tensor shape
        mock_tokenizer.encode.return_value = mock_tensor
        manager.tokenizer = mock_tokenizer
        manager.processor = Mock()
        
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = [mock_param]
        manager.model = mock_model
        manager.device = "cpu"
        
        health = manager.check_health()
        
        assert health["healthy"]
        assert health["status"] == "loaded"
        assert health["checks"]["model_loaded"]
        assert health["checks"]["tokenizer_available"]
        assert health["checks"]["processor_available"]
        assert health["checks"]["tokenizer_functional"]
        assert health["checks"]["model_on_device"]

    def test_check_health_with_cuda(self):
        """Test health check with CUDA device."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADED
        manager.device = "cuda"  # Use explicit cuda device
        
        # Mock components
        mock_tokenizer = Mock()
        mock_tensor = Mock()
        mock_tensor.shape = (1, 5)
        mock_tokenizer.encode.return_value = mock_tensor
        manager.tokenizer = mock_tokenizer
        manager.processor = Mock()
        
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = "cuda:0"
        mock_model.parameters.return_value = [mock_param]
        manager.model = mock_model
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=4 * (1024**3)), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 16 * (1024**3)  # 16GB
            
            health = manager.check_health()
            
            assert health["healthy"]
            assert health["checks"]["gpu_memory_healthy"]
            assert "gpu_memory_usage_percent" in health["checks"]

    @pytest.mark.asyncio
    async def test_get_model_info_not_loaded(self):
        """Test get_model_info when model is not loaded."""
        manager = ModelManager()
        
        with pytest.raises(ModelNotLoadedError):
            await manager.get_model_info()

    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test get_model_info when model is loaded."""
        manager = ModelManager()
        manager.status = ModelStatus.LOADED
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 50000
        manager.tokenizer = mock_tokenizer
        
        # Mock model with config
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "test_model"
        mock_model.config.max_position_embeddings = 2048
        mock_model.config.hidden_size = 768
        mock_model.config.num_attention_heads = 12
        mock_model.config.num_hidden_layers = 12
        
        # Mock parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.numel.return_value = 500
        mock_param2.requires_grad = False
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        manager.model = mock_model
        
        info = await manager.get_model_info()
        
        assert info["model_name"] == manager.model_name
        assert info["model_type"] == "test_model"
        assert info["vocab_size"] == 50000
        assert info["max_position_embeddings"] == 2048
        assert info["hidden_size"] == 768
        assert info["num_attention_heads"] == 12
        assert info["num_hidden_layers"] == 12
        assert info["total_parameters"] == 1500
        assert info["trainable_parameters"] == 1000


class TestModelManagerConcurrency:
    """Test ModelManager concurrency and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_load_model(self):
        """Test that concurrent load_model calls are handled safely."""
        manager = ModelManager()
        
        # Mock successful loading
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config = Mock()
        
        # Mock parameters properly
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        with patch('app.services.model_manager.AutoTokenizer') as mock_tokenizer_class, \
             patch('app.services.model_manager.AutoProcessor') as mock_processor_class, \
             patch('app.services.model_manager.AutoModelForCausalLM') as mock_model_class, \
             patch('app.services.model_manager.logger'):
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_processor_class.from_pretrained.return_value = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Start multiple load operations concurrently
            tasks = [manager.load_model() for _ in range(3)]
            await asyncio.gather(*tasks)
            
            # Should only load once
            assert manager.status == ModelStatus.LOADED
            assert mock_model_class.from_pretrained.call_count == 1

    @pytest.mark.asyncio
    async def test_load_unload_cycle(self):
        """Test loading and unloading model multiple times."""
        manager = ModelManager()
        
        # Mock components for loading
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config = Mock()
        
        # Mock parameters properly
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        with patch('app.services.model_manager.AutoTokenizer') as mock_tokenizer_class, \
             patch('app.services.model_manager.AutoProcessor') as mock_processor_class, \
             patch('app.services.model_manager.AutoModelForCausalLM') as mock_model_class, \
             patch('app.services.model_manager.logger'), \
             patch('torch.cuda.empty_cache'):
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_processor_class.from_pretrained.return_value = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Load -> Unload -> Load cycle
            await manager.load_model()
            assert manager.is_loaded()
            
            await manager.unload_model()
            assert not manager.is_loaded()
            
            await manager.load_model()
            assert manager.is_loaded()
            
            # Should have loaded twice
            assert mock_model_class.from_pretrained.call_count == 2