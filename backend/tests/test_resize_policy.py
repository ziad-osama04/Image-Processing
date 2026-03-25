"""Tests for ResizePolicy: target computation, resize, aspect ratio."""

import numpy as np
import pytest

from backend.domain.resize_policy import ResizeMode, ResizePolicy


class TestResizePolicyCompute:
    """Test compute_target_size for each mode."""

    def test_smallest_mode(self):
        policy = ResizePolicy(mode=ResizeMode.SMALLEST)
        target = policy.compute_target_size([(100, 200), (50, 300), (80, 150)])
        assert target == (50, 150)

    def test_largest_mode(self):
        policy = ResizePolicy(mode=ResizeMode.LARGEST)
        target = policy.compute_target_size([(100, 200), (50, 300), (80, 150)])
        assert target == (100, 300)

    def test_fixed_mode(self):
        policy = ResizePolicy(mode=ResizeMode.FIXED, fixed_width=256, fixed_height=256)
        target = policy.compute_target_size([(100, 200)])
        assert target == (256, 256)

    def test_fixed_mode_requires_dimensions(self):
        with pytest.raises(ValueError, match="requires both"):
            ResizePolicy(mode=ResizeMode.FIXED)

    def test_fixed_mode_rejects_zero(self):
        with pytest.raises(ValueError, match="requires both"):
            ResizePolicy(mode=ResizeMode.FIXED, fixed_width=0, fixed_height=256)

    def test_empty_sizes_raises(self):
        policy = ResizePolicy(mode=ResizeMode.SMALLEST)
        with pytest.raises(ValueError, match="No image sizes"):
            policy.compute_target_size([])

    def test_single_image_smallest(self):
        policy = ResizePolicy(mode=ResizeMode.SMALLEST)
        target = policy.compute_target_size([(100, 200)])
        assert target == (100, 200)


class TestResizePolicyResize:
    """Test actual image resizing."""

    def test_resize_force_stretch(self):
        img = np.random.rand(100, 200).astype(np.float64)
        result = ResizePolicy.resize_image(img, (50, 100), preserve_aspect=False)
        assert result.shape == (50, 100)
        assert result.dtype == np.float64

    def test_resize_preserve_aspect(self):
        img = np.random.rand(100, 200).astype(np.float64)
        result = ResizePolicy.resize_image(img, (100, 100), preserve_aspect=True)
        assert result.shape == (100, 100)
        assert result.dtype == np.float64

    def test_resize_same_size_returns_copy(self):
        img = np.random.rand(50, 50).astype(np.float64)
        result = ResizePolicy.resize_image(img, (50, 50))
        assert result.shape == (50, 50)
        # Should be a copy, not the same object
        assert result is not img
        np.testing.assert_array_equal(result, img)

    def test_resize_output_range(self):
        img = np.random.rand(80, 120).astype(np.float64)
        result = ResizePolicy.resize_image(img, (40, 60))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_resize_preserve_aspect_padding(self):
        # 100x200 image into 100x100 → scaled to 50x100, padded top/bottom
        img = np.ones((100, 200), dtype=np.float64)
        result = ResizePolicy.resize_image(img, (100, 100), preserve_aspect=True)
        assert result.shape == (100, 100)
        # Center should have content, edges should be padded (0)
        assert result[50, 50] > 0  # center has content
