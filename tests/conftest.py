"""Global test configuration and fixtures."""

import sys
import pytest
from unittest.mock import MagicMock


# Mock heavy libraries at the start of test session
@pytest.fixture(scope="session", autouse=True)
def mock_heavy_imports():
    """Mock heavy ML libraries to speed up tests."""
    
    # Create mock modules
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    
    # Configure torch mock
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = "cpu"
    
    # Configure transformers mock
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_processor = MagicMock()
    
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.AutoProcessor = MagicMock()
    mock_transformers.BitsAndBytesConfig = MagicMock()
    
    # Patch sys.modules
    sys.modules['torch'] = mock_torch
    sys.modules['transformers'] = mock_transformers
    sys.modules['transformers.models'] = MagicMock()
    sys.modules['transformers.models.auto'] = MagicMock()
    
    yield
    
    # Cleanup (optional, as tests usually run in isolated environments)
    for module in ['torch', 'transformers', 'transformers.models', 'transformers.models.auto']:
        if module in sys.modules:
            del sys.modules[module]