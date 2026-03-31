"""Tests for RegionMask: centered rectangle frequency-domain mask."""

import numpy as np
import pytest

from backend.domain.region_mask import RegionMask


class TestRegionMaskInner:
    """Tests for inner (low-pass) region masks."""

    def test_inner_100_percent(self) -> None:
        """Inner 100% → rectangle covers everything → all ones."""
        mask = RegionMask(size_percent=100.0, mode="inner")
        arr = mask.to_array(64, 64)
        assert arr.shape == (64, 64)
        np.testing.assert_array_equal(arr, np.ones((64, 64)))

    def test_inner_0_percent(self) -> None:
        """Inner 0% → nearly empty; only center pixel where distance=0 is included."""
        mask = RegionMask(size_percent=0.0, mode="inner")
        arr = mask.to_array(64, 64)
        # With half_h=half_w=0, only the exact center pixel (32,32)
        # has distance 0 to center (32.0, 32.0), satisfying <= 0.
        assert arr.sum() == 1.0
        assert arr[32, 32] == 1.0
        assert arr[0, 0] == 0.0

    def test_inner_50_percent(self) -> None:
        """Inner 50% → center pixels are 1, corners are 0."""
        mask = RegionMask(size_percent=50.0, mode="inner")
        arr = mask.to_array(64, 64)
        # Center pixel should be 1
        assert arr[32, 32] == 1.0
        # Corner pixels should be 0
        assert arr[0, 0] == 0.0
        assert arr[0, 63] == 0.0
        assert arr[63, 0] == 0.0
        assert arr[63, 63] == 0.0


class TestRegionMaskOuter:
    """Tests for outer (high-pass) region masks."""

    def test_outer_100_percent(self) -> None:
        """Outer 100% → rectangle covers everything → nothing outside → all zeros."""
        mask = RegionMask(size_percent=100.0, mode="outer")
        arr = mask.to_array(64, 64)
        np.testing.assert_array_equal(arr, np.zeros((64, 64)))

    def test_outer_0_percent(self) -> None:
        """Outer 0% → nearly all ones; center pixel excluded."""
        mask = RegionMask(size_percent=0.0, mode="outer")
        arr = mask.to_array(64, 64)
        # With half_h=half_w=0, only the exact center pixel (32,32)
        # is in the rectangle → outer sets it to 0.
        assert arr.sum() == 64 * 64 - 1
        assert arr[32, 32] == 0.0
        assert arr[0, 0] == 1.0

    def test_outer_50_percent(self) -> None:
        """Outer 50% → center pixels are 0, corners are 1."""
        mask = RegionMask(size_percent=50.0, mode="outer")
        arr = mask.to_array(64, 64)
        # Center pixel should be 0
        assert arr[32, 32] == 0.0
        # Corner pixels should be 1
        assert arr[0, 0] == 1.0
        assert arr[0, 63] == 1.0
        assert arr[63, 0] == 1.0
        assert arr[63, 63] == 1.0


class TestRegionMaskShape:
    """Tests for mask shape and properties."""

    def test_shape_matches_input(self) -> None:
        """Output shape must match (height, width)."""
        mask = RegionMask(size_percent=50.0, mode="inner")
        arr = mask.to_array(100, 200)
        assert arr.shape == (100, 200)

    def test_non_square_shape(self) -> None:
        """Non-square masks work correctly."""
        mask = RegionMask(size_percent=50.0, mode="inner")
        arr = mask.to_array(48, 96)
        assert arr.shape == (48, 96)
        # Center should be 1
        assert arr[24, 48] == 1.0
        # Corner should be 0
        assert arr[0, 0] == 0.0

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            RegionMask(size_percent=50.0, mode="invalid")

    def test_mask_values_binary(self) -> None:
        """All mask values should be exactly 0.0 or 1.0."""
        mask = RegionMask(size_percent=50.0, mode="inner")
        arr = mask.to_array(64, 64)
        unique_values = np.unique(arr)
        for val in unique_values:
            assert val in (0.0, 1.0)

    def test_inner_outer_complement(self) -> None:
        """Inner + Outer masks should sum to all ones."""
        inner = RegionMask(size_percent=50.0, mode="inner").to_array(64, 64)
        outer = RegionMask(size_percent=50.0, mode="outer").to_array(64, 64)
        np.testing.assert_array_equal(inner + outer, np.ones((64, 64)))
