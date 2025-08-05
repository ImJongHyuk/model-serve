"""Unit tests for model endpoint models."""

import pytest
import json
from app.schemas.model_models import ModelInfo, ModelsResponse


class TestModelInfo:
    """Test ModelInfo model."""

    def test_minimal_model_info(self):
        """Test minimal model info creation."""
        model = ModelInfo(id="gpt-3.5-turbo")
        assert model.id == "gpt-3.5-turbo"
        assert model.object == "model"
        assert model.owned_by == "huggingface"
        assert isinstance(model.created, int)
        assert model.permission == []
        assert model.root is None
        assert model.parent is None

    def test_full_model_info(self):
        """Test model info with all fields."""
        model = ModelInfo(
            id="gpt-4",
            owned_by="openai",
            root="gpt-4-base",
            parent="gpt-3.5-turbo"
        )
        assert model.id == "gpt-4"
        assert model.owned_by == "openai"
        assert model.root == "gpt-4-base"
        assert model.parent == "gpt-3.5-turbo"

    def test_model_info_serialization(self):
        """Test model info JSON serialization."""
        model = ModelInfo(id="test-model")
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        
        assert data["id"] == "test-model"
        assert data["object"] == "model"
        assert data["owned_by"] == "huggingface"
        assert "created" in data
        assert data["permission"] == []


class TestModelsResponse:
    """Test ModelsResponse model."""

    def test_empty_models_response(self):
        """Test empty models response."""
        response = ModelsResponse(data=[])
        assert response.object == "list"
        assert response.data == []

    def test_models_response_with_data(self):
        """Test models response with model data."""
        models = [
            ModelInfo(id="model-1"),
            ModelInfo(id="model-2", owned_by="custom")
        ]
        response = ModelsResponse(data=models)
        
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "model-1"
        assert response.data[1].id == "model-2"
        assert response.data[1].owned_by == "custom"

    def test_models_response_serialization(self):
        """Test models response JSON serialization."""
        models = [
            ModelInfo(id="model-1"),
            ModelInfo(id="model-2")
        ]
        response = ModelsResponse(data=models)
        
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "model-1"
        assert data["data"][1]["id"] == "model-2"