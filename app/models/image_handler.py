"""Image handling utilities for processing base64 and URL images."""

import asyncio
import base64
import io
import logging
from typing import Optional, Dict, Any, Tuple, Union
from urllib.parse import urlparse
import httpx
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from app.schemas.error_models import ImageProcessingError, ValidationError


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing and format conversion for vision models."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (336, 336),
        max_size: int = 1024,
        quality: int = 85
    ):
        """Initialize ImageProcessor.
        
        Args:
            target_size: Target size for model input (width, height)
            max_size: Maximum dimension for any side
            quality: JPEG quality for compression (1-100)
        """
        self.target_size = target_size
        self.max_size = max_size
        self.quality = quality
        
        # Define image transforms for model input
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(
        self, 
        image: Image.Image, 
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """Preprocess image for model input.
        
        Args:
            image: PIL Image object
            detail: Detail level ("low", "high", "auto")
            
        Returns:
            Dictionary containing processed image data
            
        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get original dimensions
            original_width, original_height = image.size
            
            # Determine processing based on detail level
            if detail == "low":
                # Low detail: resize to smaller size for efficiency
                processed_image = self._resize_for_low_detail(image)
            elif detail == "high":
                # High detail: preserve more detail, possibly tile large images
                processed_image = self._resize_for_high_detail(image)
            else:  # auto
                # Auto: choose based on image size
                if max(original_width, original_height) > 1024:
                    processed_image = self._resize_for_high_detail(image)
                else:
                    processed_image = self._resize_for_low_detail(image)
            
            # Apply model-specific transforms
            tensor = self.transform(processed_image)
            
            # Prepare result
            result = {
                "tensor": tensor,
                "processed_image": processed_image,
                "original_size": (original_width, original_height),
                "processed_size": processed_image.size,
                "detail_level": detail,
                "format": image.format or "UNKNOWN"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def _resize_for_low_detail(self, image: Image.Image) -> Image.Image:
        """Resize image for low detail processing.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Resized PIL Image
        """
        # Simple resize to target size
        return image.resize(self.target_size, Image.Resampling.BICUBIC)
    
    def _resize_for_high_detail(self, image: Image.Image) -> Image.Image:
        """Resize image for high detail processing.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        
        # If image is too large, resize while maintaining aspect ratio
        if max(width, height) > self.max_size:
            if width > height:
                new_width = self.max_size
                new_height = int(height * (self.max_size / width))
            else:
                new_height = self.max_size
                new_width = int(width * (self.max_size / height))
            
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Then resize to target size for model
        return image.resize(self.target_size, Image.Resampling.BICUBIC)
    
    def optimize_image_size(self, image: Image.Image) -> Image.Image:
        """Optimize image size for processing efficiency.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Optimized PIL Image
        """
        width, height = image.size
        
        # Calculate total pixels
        total_pixels = width * height
        
        # If image is very large, reduce size
        if total_pixels > 1024 * 1024:  # > 1MP
            # Calculate scale factor to reduce to ~1MP
            scale_factor = (1024 * 1024 / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        return image


class ImageHandler:
    """Handles image loading, processing, and format conversion."""
    
    def __init__(
        self,
        max_image_size: int = 10 * 1024 * 1024,  # 10MB
        timeout: int = 30,
        max_concurrent_downloads: int = 5
    ):
        """Initialize ImageHandler.
        
        Args:
            max_image_size: Maximum image size in bytes
            timeout: HTTP request timeout in seconds
            max_concurrent_downloads: Maximum concurrent image downloads
        """
        self.max_image_size = max_image_size
        self.timeout = timeout
        self.max_concurrent_downloads = max_concurrent_downloads
        self.processor = ImageProcessor()
        
        # HTTP client for URL fetching
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_concurrent_downloads)
        )
        
        # Supported image formats
        self.supported_formats = {
            'JPEG', 'JPG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP'
        }
    
    async def process_image_url(
        self, 
        url: str, 
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """Process image from URL.
        
        Args:
            url: Image URL (http/https or data URI)
            detail: Detail level for processing
            
        Returns:
            Dictionary containing processed image data
            
        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            if url.startswith('data:'):
                # Handle data URI (base64)
                return await self._process_data_uri(url, detail)
            else:
                # Handle HTTP/HTTPS URL
                return await self._process_http_url(url, detail)
                
        except Exception as e:
            logger.error(f"Error processing image URL {url}: {e}")
            raise ImageProcessingError(f"Failed to process image from URL: {str(e)}")
    
    async def _process_data_uri(self, data_uri: str, detail: str) -> Dict[str, Any]:
        """Process image from data URI.
        
        Args:
            data_uri: Data URI string
            detail: Detail level for processing
            
        Returns:
            Dictionary containing processed image data
        """
        try:
            # Parse data URI
            if not data_uri.startswith('data:image/'):
                raise ValueError("Invalid data URI format")
            
            # Extract base64 data
            header, data = data_uri.split(',', 1)
            
            # Decode base64
            image_data = base64.b64decode(data)
            
            # Check size
            if len(image_data) > self.max_image_size:
                raise ValueError(f"Image size ({len(image_data)} bytes) exceeds maximum ({self.max_image_size} bytes)")
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Validate format
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image.format}")
            
            # Process image
            processed = self.processor.preprocess_image(image, detail)
            processed.update({
                "source": "data_uri",
                "url": data_uri[:100] + "..." if len(data_uri) > 100 else data_uri,
                "size_bytes": len(image_data)
            })
            
            return processed
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to process data URI image: {str(e)}")
    
    async def _process_http_url(self, url: str, detail: str) -> Dict[str, Any]:
        """Process image from HTTP/HTTPS URL.
        
        Args:
            url: HTTP/HTTPS URL
            detail: Detail level for processing
            
        Returns:
            Dictionary containing processed image data
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
            
            # Download image
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"Invalid content type: {content_type}")
            
            # Check size
            image_data = response.content
            if len(image_data) > self.max_image_size:
                raise ValueError(f"Image size ({len(image_data)} bytes) exceeds maximum ({self.max_image_size} bytes)")
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Validate format
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image.format}")
            
            # Process image
            processed = self.processor.preprocess_image(image, detail)
            processed.update({
                "source": "http_url",
                "url": url,
                "size_bytes": len(image_data),
                "content_type": content_type
            })
            
            return processed
            
        except httpx.HTTPError as e:
            raise ImageProcessingError(f"HTTP error downloading image: {str(e)}")
        except Exception as e:
            raise ImageProcessingError(f"Failed to process HTTP image: {str(e)}")
    
    async def process_multiple_images(
        self, 
        image_urls: list[str], 
        detail: str = "auto"
    ) -> list[Dict[str, Any]]:
        """Process multiple images concurrently.
        
        Args:
            image_urls: List of image URLs
            detail: Detail level for processing
            
        Returns:
            List of processed image data dictionaries
        """
        if not image_urls:
            return []
        
        # Limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def process_single(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.process_image_url(url, detail)
                except Exception as e:
                    logger.error(f"Failed to process image {url}: {e}")
                    return {
                        "error": str(e),
                        "url": url,
                        "source": "error"
                    }
        
        # Process all images concurrently
        tasks = [process_single(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        processed_images = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Image processing exception: {result}")
                continue
            if "error" not in result:
                processed_images.append(result)
        
        return processed_images
    
    def validate_image_format(self, image: Image.Image) -> bool:
        """Validate if image format is supported.
        
        Args:
            image: PIL Image object
            
        Returns:
            True if format is supported
        """
        return image.format in self.supported_formats
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get detailed information about an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image information
        """
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            "info": dict(image.info)
        }
    
    async def close(self):
        """Close HTTP client and cleanup resources."""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Utility functions
def is_data_uri(url: str) -> bool:
    """Check if URL is a data URI.
    
    Args:
        url: URL string
        
    Returns:
        True if URL is a data URI
    """
    return url.startswith('data:')


def is_valid_image_url(url: str) -> bool:
    """Validate if URL could be a valid image URL.
    
    Args:
        url: URL string
        
    Returns:
        True if URL format is valid
    """
    if is_data_uri(url):
        return url.startswith('data:image/')
    
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def estimate_image_tokens(width: int, height: int, detail: str = "auto") -> int:
    """Estimate token count for image processing.
    
    Args:
        width: Image width
        height: Image height
        detail: Detail level
        
    Returns:
        Estimated token count
    """
    # Base token cost for image processing
    base_tokens = 85
    
    if detail == "low":
        return base_tokens
    elif detail == "auto":
        # Auto: choose based on image size
        if max(width, height) <= 1024:
            return base_tokens  # Use low detail for small images
    
    # For high detail, calculate based on image tiles
    # Each 512x512 tile costs additional tokens
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    total_tiles = tiles_x * tiles_y
    
    # Additional tokens per tile
    tile_tokens = 170
    
    return base_tokens + (total_tiles * tile_tokens)


def create_image_placeholder(width: int, height: int, color: str = "gray") -> Image.Image:
    """Create a placeholder image for testing.
    
    Args:
        width: Image width
        height: Image height
        color: Placeholder color
        
    Returns:
        PIL Image placeholder
    """
    return Image.new('RGB', (width, height), color)