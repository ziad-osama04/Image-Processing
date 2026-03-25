"""Tests for FourierData: FFT, components, round-trip fidelity."""

import numpy as np
import pytest

from backend.domain.fourier_data import FourierData


def _make_test_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a simple test image (float64, 0-1)."""
    rng = np.random.default_rng(42)
    return rng.random((h, w))


class TestFourierDataConstruction:
    """Test FFT computation and spectrum shape."""

    def test_from_image_shape(self):
        img = _make_test_image(64, 64)
        fd = FourierData.from_image(img)
        assert fd.spectrum.shape == (64, 64)
        assert fd.spectrum.dtype == np.complex128

    def test_from_image_non_square(self):
        img = _make_test_image(48, 96)
        fd = FourierData.from_image(img)
        assert fd.spectrum.shape == (48, 96)

    def test_from_image_non_power_of_2(self):
        img = _make_test_image(50, 73)
        fd = FourierData.from_image(img)
        assert fd.spectrum.shape == (50, 73)


class TestFourierDataComponents:
    """Test component extraction and caching."""

    def test_magnitude_uint8(self):
        fd = FourierData.from_image(_make_test_image())
        mag = fd.magnitude
        assert mag.dtype == np.uint8
        assert mag.shape == (64, 64)

    def test_phase_uint8(self):
        fd = FourierData.from_image(_make_test_image())
        ph = fd.phase
        assert ph.dtype == np.uint8

    def test_real_uint8(self):
        fd = FourierData.from_image(_make_test_image())
        r = fd.real_part
        assert r.dtype == np.uint8

    def test_imaginary_uint8(self):
        fd = FourierData.from_image(_make_test_image())
        im = fd.imaginary_part
        assert im.dtype == np.uint8

    def test_components_are_distinct(self):
        fd = FourierData.from_image(_make_test_image())
        mag = fd.magnitude
        phase = fd.phase
        real = fd.real_part
        imag = fd.imaginary_part
        # At least some should differ
        assert not (np.array_equal(mag, phase) and np.array_equal(real, imag))

    def test_caching(self):
        fd = FourierData.from_image(_make_test_image())
        mag1 = fd.magnitude
        mag2 = fd.magnitude
        assert mag1 is mag2  # same object, cached

    def test_get_component_valid(self):
        fd = FourierData.from_image(_make_test_image())
        for name in ("magnitude", "phase", "real", "imaginary"):
            comp = fd.get_component(name)
            assert comp.dtype == np.uint8

    def test_get_component_invalid(self):
        fd = FourierData.from_image(_make_test_image())
        with pytest.raises(ValueError, match="Invalid component"):
            fd.get_component("invalid")


class TestFourierDataDisplay:
    """Test display with B/C."""

    def test_display_default_bc(self):
        fd = FourierData.from_image(_make_test_image())
        display = fd.get_display_component("magnitude")
        assert display.dtype == np.uint8

    def test_display_brightness(self):
        fd = FourierData.from_image(_make_test_image())
        default = fd.get_display_component("magnitude", brightness=0.0)
        bright = fd.get_display_component("magnitude", brightness=0.5)
        assert bright.mean() >= default.mean()

    def test_display_clamped(self):
        fd = FourierData.from_image(_make_test_image())
        display = fd.get_display_component("magnitude", brightness=1.0, contrast=10.0)
        assert display.max() <= 255
        assert display.min() >= 0


class TestFourierDataRoundTrip:
    """Test FFT→IFFT reconstruction fidelity."""

    @pytest.mark.parametrize("shape", [(64, 64), (50, 73), (128, 128)])
    def test_roundtrip_fidelity(self, shape):
        """Round-trip error must be ≤ 1e-6 (FR-007)."""
        img = _make_test_image(*shape)
        fd = FourierData.from_image(img)
        reconstructed = fd.reconstruct()
        max_error = np.max(np.abs(img - reconstructed))
        assert max_error < 1e-6, f"Round-trip error {max_error} exceeds 1e-6"

    def test_roundtrip_single_pixel(self):
        """FT of a single pixel is itself."""
        img = np.array([[0.5]], dtype=np.float64)
        fd = FourierData.from_image(img)
        reconstructed = fd.reconstruct()
        np.testing.assert_allclose(reconstructed, img, atol=1e-10)
