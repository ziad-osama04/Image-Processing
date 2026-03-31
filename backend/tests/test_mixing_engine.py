"""Tests for MixingEngine: weighted FT component mixing."""

import numpy as np
import pytest

from backend.domain.image_model import ImageModel
from backend.domain.mixing_engine import MixingEngine
from backend.domain.region_mask import RegionMask


def _make_image_model(h: int = 64, w: int = 64, seed: int = 42) -> ImageModel:
    """Create an ImageModel with a random grayscale image."""
    rng = np.random.default_rng(seed)
    img = ImageModel()
    img.grayscale = rng.random((h, w))
    img.filename = f"test_{seed}.png"
    return img


class TestMixingEngineSingleImage:
    """Tests for single-image mixing (identity-like operations)."""

    def test_single_image_identity_real_imag(self) -> None:
        """Mixing 1 image with weights (1, 1) in real-imag mode ≈ original."""
        img = _make_image_model(64, 64, seed=42)
        original = img.get_active_array()

        result = MixingEngine.mix(
            images=[img, None, None, None],
            mode="real-imag",
            weights_a=[1.0, 0.0, 0.0, 0.0],
            weights_b=[1.0, 0.0, 0.0, 0.0],
            mask=None,
        )

        assert result.shape == original.shape
        assert result.dtype == np.float64
        # After normalization the shape of the result should match
        # The reconstruction won't be bit-identical due to normalization,
        # but correlation should be very high
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # The result should approximate the original (up to normalization)
        np.testing.assert_allclose(result, original, atol=1e-3)

    def test_single_image_identity_mag_phase(self) -> None:
        """Mixing 1 image with weights (1, 1) in mag-phase mode ≈ original."""
        img = _make_image_model(64, 64, seed=42)
        original = img.get_active_array()

        result = MixingEngine.mix(
            images=[img, None, None, None],
            mode="mag-phase",
            weights_a=[1.0, 0.0, 0.0, 0.0],
            weights_b=[1.0, 0.0, 0.0, 0.0],
            mask=None,
        )

        assert result.shape == original.shape
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, original, atol=1e-3)


class TestMixingEngineZeroWeights:
    """Tests for zero-weight edge cases."""

    def test_all_zero_weights(self) -> None:
        """All-zero weights → output is all zeros."""
        img = _make_image_model(64, 64)

        result = MixingEngine.mix(
            images=[img, None, None, None],
            mode="real-imag",
            weights_a=[0.0, 0.0, 0.0, 0.0],
            weights_b=[0.0, 0.0, 0.0, 0.0],
            mask=None,
        )

        # With all-zero weights, result should be all zeros
        np.testing.assert_array_equal(result, np.zeros((64, 64)))


class TestMixingEngineMultiImage:
    """Tests for mixing multiple images."""

    def test_mag_phase_mode_two_images(self) -> None:
        """Mixing 2 images in mag-phase mode produces valid output."""
        img1 = _make_image_model(64, 64, seed=1)
        img2 = _make_image_model(64, 64, seed=2)

        result = MixingEngine.mix(
            images=[img1, img2, None, None],
            mode="mag-phase",
            weights_a=[1.0, 0.0, 0.0, 0.0],
            weights_b=[0.0, 1.0, 0.0, 0.0],
            mask=None,
        )

        assert result.shape == (64, 64)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_real_imag_mode_two_images(self) -> None:
        """Mixing 2 images in real-imag mode produces valid output."""
        img1 = _make_image_model(64, 64, seed=1)
        img2 = _make_image_model(64, 64, seed=2)

        result = MixingEngine.mix(
            images=[img1, img2, None, None],
            mode="real-imag",
            weights_a=[0.5, 0.5, 0.0, 0.0],
            weights_b=[0.5, 0.5, 0.0, 0.0],
            mask=None,
        )

        assert result.shape == (64, 64)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestMixingEngineWithMask:
    """Tests for mixing with region masks."""

    def test_inner_mask_changes_result(self) -> None:
        """Inner 10% mask produces a different result than no mask."""
        img = _make_image_model(64, 64, seed=42)

        # No mask (equivalent to inner 100%)
        result_no_mask = MixingEngine.mix(
            images=[img, None, None, None],
            mode="real-imag",
            weights_a=[1.0, 0.0, 0.0, 0.0],
            weights_b=[1.0, 0.0, 0.0, 0.0],
            mask=None,
        )

        # Inner 10% mask
        mask = RegionMask(size_percent=10.0, mode="inner").to_array(64, 64)
        result_masked = MixingEngine.mix(
            images=[img, None, None, None],
            mode="real-imag",
            weights_a=[1.0, 0.0, 0.0, 0.0],
            weights_b=[1.0, 0.0, 0.0, 0.0],
            mask=mask,
        )

        # The masked result should be different from unmasked
        assert not np.array_equal(result_no_mask, result_masked)


class TestMixingEngineEdgeCases:
    """Tests for edge cases."""

    def test_no_loaded_images(self) -> None:
        """Passing all-None images returns zero array of default shape."""
        result = MixingEngine.mix(
            images=[None, None, None, None],
            mode="real-imag",
            weights_a=[0.0, 0.0, 0.0, 0.0],
            weights_b=[0.0, 0.0, 0.0, 0.0],
            mask=None,
        )

        assert result.shape == (256, 256)
        np.testing.assert_array_equal(result, np.zeros((256, 256)))

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        img = _make_image_model(64, 64)
        with pytest.raises(ValueError, match="Invalid mode"):
            MixingEngine.mix(
                images=[img],
                mode="invalid",
                weights_a=[1.0],
                weights_b=[1.0],
                mask=None,
            )
