"""Unit tests for ImageHandler and ImageProcessor classes."""

import pytest
import asyncio
import base64
import io
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import torch
import httpx

from app.models.image_handler import (
    ImageHandler,
    ImageProcessor,
    is_data_uri,
    is_valid_image_url,
    estimate_image_tokens,
    create_image_placeholder
)
from app.schemas.error_models import ImageProcessingError


class TestImageProcessor:
    """Test ImageProcessor class."""

    def test_init_default(self):
        """Test ImageProcessor initialization with defaults."""
        processor = ImageProcessor()
        assert processor.target_size == (336, 336)
        assert processor.max_size == 1024
        assert processor.quality == 85
        assert processor.transform is not None

    def test_init_custom(self):
        """Test ImageProcessor initialization with custom values."""
        processor = ImageProcessor(
            target_size=(512, 512),
            max_size=2048,
            quality=95
        )
        assert processor.target_size == (512, 512)
        assert processor.max_size == 2048
        assert processor.quality == 95

    def test_preprocess_image_rgb(self):
        """Test preprocessing RGB image."""
        processor = ImageProcessor()
        
        # Create test image
        image = Image.new('RGB', (100, 100), 'red')
        
        result = processor.preprocess_image(image, detail="auto")
        
        assert "tensor" in result
        assert "processed_image" in result
        assert "original_size" in result
        assert "processed_size" in result
        assert result["original_size"] == (100, 100)
        assert result["processed_size"] == (336, 336)
        assert result["detail_level"] == "auto"
        
        # Check tensor shape
        tensor = result["tensor"]
        assert tensor.shape == (3, 336, 336)  # C, H, W

    def test_preprocess_image_rgba_conversion(self):
        """Test preprocessing RGBA image (should convert to RGB)."""
        processor = ImageProcessor()
        
        # Create RGBA image
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        
        result = processor.preprocess_image(image, detail="auto")
        
        # Should be converted to RGB
        assert result["processed_image"].mode == 'RGB'

    def test_preprocess_image_low_detail(self):
        """Test preprocessing with low detail."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (1000, 1000), 'blue')
        
        result = processor.preprocess_image(image, detail="low")
        
        assert result["detail_level"] == "low"
        assert result["processed_size"] == (336, 336)

    def test_preprocess_image_high_detail(self):
        """Test preprocessing with high detail."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (2000, 2000), 'green')
        
        result = processor.preprocess_image(image, detail="high")
        
        assert result["detail_level"] == "high"
        assert result["processed_size"] == (336, 336)
        # Original large image should be resized to max_size first
        assert result["original_size"] == (2000, 2000)

    def test_preprocess_image_auto_detail_small(self):
        """Test preprocessing with auto detail for small image."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (500, 500), 'yellow')
        
        result = processor.preprocess_image(image, detail="auto")
        
        # Should use low detail for small images
        assert result["detail_level"] == "auto"

    def test_preprocess_image_auto_detail_large(self):
        """Test preprocessing with auto detail for large image."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (1500, 1500), 'purple')
        
        result = processor.preprocess_image(image, detail="auto")
        
        # Should use high detail for large images
        assert result["detail_level"] == "auto"

    def test_resize_for_low_detail(self):
        """Test low detail resizing."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (1000, 500), 'red')
        resized = processor._resize_for_low_detail(image)
        
        assert resized.size == (336, 336)

    def test_resize_for_high_detail_large_image(self):
        """Test high detail resizing for large image."""
        processor = ImageProcessor(max_size=1024)
        
        image = Image.new('RGB', (2000, 1000), 'blue')
        resized = processor._resize_for_high_detail(image)
        
        # Should be resized to target size
        assert resized.size == (336, 336)

    def test_resize_for_high_detail_small_image(self):
        """Test high detail resizing for small image."""
        processor = ImageProcessor()
        
        image = Image.new('RGB', (500, 300), 'green')
        resized = processor._resize_for_high_detail(image)
        
        # Should be resized to target size
        assert resized.size == (336, 336)

    def test_optimize_image_size_large(self):
        """Test image size optimization for large image."""
        processor = ImageProcessor()
        
        # Create large image (> 1MP)
        image = Image.new('RGB', (2000, 2000), 'red')  # 4MP
        optimized = processor.optimize_image_size(image)
        
        # Should be reduced to ~1MP
        total_pixels = optimized.width * optimized.height
        assert total_pixels <= 1024 * 1024 * 1.1  # Allow some margin

    def test_optimize_image_size_small(self):
        """Test image size optimization for small image."""
        processor = ImageProcessor()
        
        # Create small image (< 1MP)
        image = Image.new('RGB', (500, 500), 'blue')  # 0.25MP
        optimized = processor.optimize_image_size(image)
        
        # Should remain unchanged
        assert optimized.size == (500, 500)

    def test_preprocess_image_error_handling(self):
        """Test error handling in image preprocessing."""
        processor = ImageProcessor()
        
        # Mock an error in the transform
        with patch.object(processor, 'transform', side_effect=Exception("Transform error")):
            with pytest.raises(ImageProcessingError):
                processor.preprocess_image(Image.new('RGB', (100, 100), 'red'))


class TestImageHandler:
    """Test ImageHandler class."""

    def test_init_default(self):
        """Test ImageHandler initialization with defaults."""
        handler = ImageHandler()
        assert handler.max_image_size == 10 * 1024 * 1024
        assert handler.timeout == 30
        assert handler.max_concurrent_downloads == 5
        assert isinstance(handler.processor, ImageProcessor)
        assert handler.http_client is not None

    def test_init_custom(self):
        """Test ImageHandler initialization with custom values."""
        handler = ImageHandler(
            max_image_size=5 * 1024 * 1024,
            timeout=60,
            max_concurrent_downloads=10
        )
        assert handler.max_image_size == 5 * 1024 * 1024
        assert handler.timeout == 60
        assert handler.max_concurrent_downloads == 10

    @pytest.mark.asyncio
    async def test_process_data_uri_valid(self):
        """Test processing valid data URI."""
        handler = ImageHandler()
        
        # Create test image and convert to base64
        test_image = Image.new('RGB', (100, 100), 'red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Create data URI
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_data}"
        
        result = await handler._process_data_uri(data_uri, "auto")
        
        assert result["source"] == "data_uri"
        assert result["size_bytes"] == len(image_data)
        assert "tensor" in result
        assert "processed_image" in result

    @pytest.mark.asyncio
    async def test_process_data_uri_invalid_format(self):
        """Test processing data URI with invalid format."""
        handler = ImageHandler()
        
        invalid_uri = "data:text/plain;base64,SGVsbG8gV29ybGQ="
        
        with pytest.raises(ImageProcessingError):
            await handler._process_data_uri(invalid_uri, "auto")

    @pytest.mark.asyncio
    async def test_process_data_uri_invalid_base64(self):
        """Test processing data URI with invalid base64."""
        handler = ImageHandler()
        
        invalid_uri = "data:image/png;base64,invalid_base64_data"
        
        with pytest.raises(ImageProcessingError):
            await handler._process_data_uri(invalid_uri, "auto")

    @pytest.mark.asyncio
    async def test_process_data_uri_too_large(self):
        """Test processing data URI that's too large."""
        handler = ImageHandler(max_image_size=100)  # Very small limit
        
        # Create large image
        test_image = Image.new('RGB', (1000, 1000), 'red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_data}"
        
        with pytest.raises(ImageProcessingError):
            await handler._process_data_uri(data_uri, "auto")

    @pytest.mark.asyncio
    async def test_process_http_url_success(self):
        """Test processing HTTP URL successfully."""
        handler = ImageHandler()
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), 'blue')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.raise_for_status = Mock()
        
        with patch.object(handler.http_client, 'get', return_value=mock_response):
            result = await handler._process_http_url("https://example.com/image.jpg", "auto")
            
            assert result["source"] == "http_url"
            assert result["url"] == "https://example.com/image.jpg"
            assert result["size_bytes"] == len(image_data)
            assert result["content_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_process_http_url_invalid_scheme(self):
        """Test processing HTTP URL with invalid scheme."""
        handler = ImageHandler()
        
        with pytest.raises(ImageProcessingError):
            await handler._process_http_url("ftp://example.com/image.jpg", "auto")

    @pytest.mark.asyncio
    async def test_process_http_url_http_error(self):
        """Test processing HTTP URL with HTTP error."""
        handler = ImageHandler()
        
        with patch.object(handler.http_client, 'get', side_effect=httpx.HTTPError("Network error")):
            with pytest.raises(ImageProcessingError):
                await handler._process_http_url("https://example.com/image.jpg", "auto")

    @pytest.mark.asyncio
    async def test_process_http_url_invalid_content_type(self):
        """Test processing HTTP URL with invalid content type."""
        handler = ImageHandler()
        
        mock_response = Mock()
        mock_response.content = b"not an image"
        mock_response.headers = {'content-type': 'text/plain'}
        mock_response.raise_for_status = Mock()
        
        with patch.object(handler.http_client, 'get', return_value=mock_response):
            with pytest.raises(ImageProcessingError):
                await handler._process_http_url("https://example.com/text.txt", "auto")

    @pytest.mark.asyncio
    async def test_process_image_url_data_uri(self):
        """Test processing image URL that's a data URI."""
        handler = ImageHandler()
        
        # Create test data URI
        test_image = Image.new('RGB', (50, 50), 'green')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_data}"
        
        with patch.object(handler, '_process_data_uri', return_value={"test": "data"}) as mock_process:
            result = await handler.process_image_url(data_uri, "high")
            
            mock_process.assert_called_once_with(data_uri, "high")
            assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_process_image_url_http(self):
        """Test processing image URL that's HTTP."""
        handler = ImageHandler()
        
        with patch.object(handler, '_process_http_url', return_value={"test": "http"}) as mock_process:
            result = await handler.process_image_url("https://example.com/image.jpg", "low")
            
            mock_process.assert_called_once_with("https://example.com/image.jpg", "low")
            assert result == {"test": "http"}

    @pytest.mark.asyncio
    async def test_process_multiple_images_empty(self):
        """Test processing empty list of images."""
        handler = ImageHandler()
        
        result = await handler.process_multiple_images([])
        assert result == []

    @pytest.mark.asyncio
    async def test_process_multiple_images_success(self):
        """Test processing multiple images successfully."""
        handler = ImageHandler()
        
        urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
        
        with patch.object(handler, 'process_image_url', side_effect=[
            {"url": "1.jpg", "success": True},
            {"url": "2.jpg", "success": True}
        ]):
            results = await handler.process_multiple_images(urls)
            
            assert len(results) == 2
            assert results[0]["url"] == "1.jpg"
            assert results[1]["url"] == "2.jpg"

    @pytest.mark.asyncio
    async def test_process_multiple_images_with_errors(self):
        """Test processing multiple images with some errors."""
        handler = ImageHandler()
        
        urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
        
        with patch.object(handler, 'process_image_url', side_effect=[
            {"url": "1.jpg", "success": True},
            Exception("Failed to process")
        ]):
            results = await handler.process_multiple_images(urls)
            
            # Should only return successful results
            assert len(results) == 1
            assert results[0]["url"] == "1.jpg"

    def test_validate_image_format_supported(self):
        """Test validating supported image format."""
        handler = ImageHandler()
        
        image = Image.new('RGB', (100, 100), 'red')
        image.format = 'JPEG'
        
        assert handler.validate_image_format(image) is True

    def test_validate_image_format_unsupported(self):
        """Test validating unsupported image format."""
        handler = ImageHandler()
        
        image = Image.new('RGB', (100, 100), 'red')
        image.format = 'UNSUPPORTED'
        
        assert handler.validate_image_format(image) is False

    def test_get_image_info(self):
        """Test getting image information."""
        handler = ImageHandler()
        
        image = Image.new('RGBA', (200, 150), (255, 0, 0, 128))
        image.format = 'PNG'
        
        info = handler.get_image_info(image)
        
        assert info["format"] == "PNG"
        assert info["mode"] == "RGBA"
        assert info["size"] == (200, 150)
        assert info["width"] == 200
        assert info["height"] == 150
        assert info["has_transparency"] is True

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using ImageHandler as async context manager."""
        async with ImageHandler() as handler:
            assert isinstance(handler, ImageHandler)
            assert handler.http_client is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing ImageHandler."""
        handler = ImageHandler()
        
        with patch.object(handler.http_client, 'aclose', new_callable=AsyncMock) as mock_close:
            await handler.close()
            mock_close.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_data_uri_true(self):
        """Test is_data_uri with valid data URI."""
        assert is_data_uri("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==") is True

    def test_is_data_uri_false(self):
        """Test is_data_uri with regular URL."""
        assert is_data_uri("https://example.com/image.jpg") is False

    def test_is_valid_image_url_data_uri(self):
        """Test is_valid_image_url with data URI."""
        valid_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert is_valid_image_url(valid_data_uri) is True
        
        invalid_data_uri = "data:text/plain;base64,SGVsbG8="
        assert is_valid_image_url(invalid_data_uri) is False

    def test_is_valid_image_url_http(self):
        """Test is_valid_image_url with HTTP URLs."""
        assert is_valid_image_url("https://example.com/image.jpg") is True
        assert is_valid_image_url("http://example.com/image.png") is True
        assert is_valid_image_url("ftp://example.com/image.jpg") is False
        assert is_valid_image_url("invalid-url") is False

    def test_estimate_image_tokens_low_detail(self):
        """Test estimating tokens for low detail image."""
        tokens = estimate_image_tokens(1000, 1000, "low")
        assert tokens == 85  # Base tokens only

    def test_estimate_image_tokens_high_detail(self):
        """Test estimating tokens for high detail image."""
        tokens = estimate_image_tokens(1024, 1024, "high")
        # Should be base + tile tokens
        # 1024x1024 = 2x2 tiles = 4 tiles
        expected = 85 + (4 * 170)
        assert tokens == expected

    def test_estimate_image_tokens_auto_small(self):
        """Test estimating tokens for auto detail small image."""
        tokens = estimate_image_tokens(500, 500, "auto")
        assert tokens == 85  # Should use low detail

    def test_estimate_image_tokens_auto_large(self):
        """Test estimating tokens for auto detail large image."""
        tokens = estimate_image_tokens(2000, 1000, "auto")
        # Should calculate tiles for high detail
        tiles_x = (2000 + 511) // 512  # 4 tiles
        tiles_y = (1000 + 511) // 512  # 2 tiles
        total_tiles = tiles_x * tiles_y  # 8 tiles
        expected = 85 + (total_tiles * 170)
        assert tokens == expected

    def test_create_image_placeholder(self):
        """Test creating image placeholder."""
        placeholder = create_image_placeholder(100, 200, "red")
        
        assert isinstance(placeholder, Image.Image)
        assert placeholder.size == (100, 200)
        assert placeholder.mode == "RGB"


class TestImageHandlerIntegration:
    """Integration tests for ImageHandler."""

    @pytest.mark.asyncio
    async def test_full_data_uri_processing_flow(self):
        """Test complete data URI processing flow."""
        handler = ImageHandler()
        
        # Create test image
        test_image = Image.new('RGB', (200, 200), 'blue')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Create data URI
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_data}"
        
        # Process image
        result = await handler.process_image_url(data_uri, "high")
        
        # Verify results
        assert result["source"] == "data_uri"
        assert result["detail_level"] == "high"
        assert result["original_size"] == (200, 200)
        assert result["processed_size"] == (336, 336)
        assert "tensor" in result
        
        # Verify tensor properties
        tensor = result["tensor"]
        assert tensor.shape == (3, 336, 336)
        assert tensor.dtype == torch.float32

    @pytest.mark.asyncio
    async def test_error_handling_flow(self):
        """Test error handling throughout the processing flow."""
        handler = ImageHandler()
        
        # Test various error conditions
        error_cases = [
            "invalid-url",
            "data:text/plain;base64,SGVsbG8=",
            "https://",
            "data:image/png;base64,invalid_base64"
        ]
        
        for url in error_cases:
            with pytest.raises(ImageProcessingError):
                await handler.process_image_url(url, "auto")

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent image processing."""
        handler = ImageHandler()
        
        # Create multiple test images
        test_images = []
        colors = ['red', 'green', 'blue']
        for i in range(3):
            test_image = Image.new('RGB', (50, 50), colors[i])
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_data}"
            test_images.append(data_uri)
        
        # Process concurrently
        results = await handler.process_multiple_images(test_images, "auto")
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["source"] == "data_uri"
            assert "tensor" in result