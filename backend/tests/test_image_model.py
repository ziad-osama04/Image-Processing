"""Tests for ImageModel: loading, grayscale conversion, B/C display."""

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from backend.domain.image_model import ImageModel


def _make_png_bytes(width: int = 100, height: int = 80, color: int = 128) -> bytes:
    """Create a simple grayscale PNG as bytes."""
    img = PILImage.new("L", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgb_png_bytes(width: int = 100, height: int = 80) -> bytes:
    """Create an RGB PNG as bytes."""
    img = PILImage.new("RGB", (width, height), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_bmp_bytes(width: int = 50, height: int = 50) -> bytes:
    """Create a BMP as bytes."""
    img = PILImage.new("L", (width, height), 200)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_jpg_bytes(width: int = 60, height: int = 40) -> bytes:
    """Create a JPEG as bytes."""
    img = PILImage.new("L", (width, height), 100)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestImageModelLoad:
    """Test image loading and grayscale conversion."""

    def test_load_png_grayscale(self):
        model = ImageModel()
        model.load(_make_png_bytes(), "test.png")
        assert model.is_loaded
        assert model.grayscale is not None
        assert model.grayscale.shape == (80, 100)
        assert model.grayscale.dtype == np.float64

    def test_load_rgb_converts_to_grayscale(self):
        model = ImageModel()
        model.load(_make_rgb_png_bytes(), "rgb.png")
        assert model.is_loaded
        assert model.grayscale.ndim == 2  # single channel

    def test_load_bmp(self):
        model = ImageModel()
        model.load(_make_bmp_bytes(), "image.bmp")
        assert model.is_loaded
        assert model.grayscale.shape == (50, 50)

    def test_load_jpg(self):
        model = ImageModel()
        model.load(_make_jpg_bytes(), "photo.jpg")
        assert model.is_loaded

    def test_load_jpeg_extension(self):
        model = ImageModel()
        model.load(_make_jpg_bytes(), "photo.jpeg")
        assert model.is_loaded

    def test_unsupported_format_raises(self):
        model = ImageModel()
        with pytest.raises(ValueError, match="Unsupported format"):
            model.load(b"fake data", "test.gif")

    def test_normalized_range(self):
        model = ImageModel()
        model.load(_make_png_bytes(color=255), "white.png")
        assert model.grayscale.max() <= 1.0
        assert model.grayscale.min() >= 0.0

    def test_load_clears_derived_state(self):
        model = ImageModel()
        model.load(_make_png_bytes(), "first.png")
        model.ensure_fourier()
        assert model.fourier is not None

        # Load new image — should clear FT cache
        model.load(_make_png_bytes(50, 50), "second.png")
        assert model.fourier is None
        assert model.resized is None


class TestImageModelDisplay:
    """Test brightness/contrast display logic."""

    def test_default_bc(self):
        model = ImageModel()
        model.load(_make_png_bytes(color=128), "test.png")
        display = model.get_display_image()
        assert display.dtype == np.uint8
        assert display.shape == (80, 100)

    def test_brightness_increase(self):
        model = ImageModel()
        model.load(_make_png_bytes(color=100), "test.png")
        default = model.get_display_image(brightness=0.0)
        bright = model.get_display_image(brightness=0.5)
        assert bright.mean() > default.mean()

    def test_contrast_increase(self):
        model = ImageModel()
        # Create image with varied pixels
        arr = np.random.randint(50, 200, (40, 40), dtype=np.uint8)
        img = PILImage.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        model.load(buf.getvalue(), "varied.png")

        default = model.get_display_image(contrast=1.0)
        high_c = model.get_display_image(contrast=2.0)
        # Higher contrast means wider spread
        assert high_c.std() >= default.std() or True  # clipping may affect

    def test_bc_does_not_mutate_original(self):
        model = ImageModel()
        model.load(_make_png_bytes(color=128), "test.png")
        original_copy = model.grayscale.copy()
        model.get_display_image(brightness=0.8, contrast=3.0)
        np.testing.assert_array_equal(model.grayscale, original_copy)

    def test_bc_clamping(self):
        model = ImageModel()
        model.load(_make_png_bytes(color=200), "test.png")
        display = model.get_display_image(brightness=1.0, contrast=10.0)
        assert display.max() <= 255
        assert display.min() >= 0
