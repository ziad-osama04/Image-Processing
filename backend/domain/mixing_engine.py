"""
MixingEngine: Weighted FT component mixing across multiple images.

All mixing math lives here — no math in routers or frontend.
Uses numpy vectorized operations for performance (Constitution §II).
"""

import numpy as np

from backend.domain.image_model import ImageModel


class MixingEngine:
    """
    Mixes frequency-domain components of multiple images using weighted averages.

    Supports two mixing modes:
    - 'mag-phase': mix magnitudes and phases separately, reconstruct via polar form.
    - 'real-imag': mix real and imaginary parts separately, combine directly.
    """

    @staticmethod
    def mix(
        images: list[ImageModel | None],
        mode: str,
        weights_a: list[float],
        weights_b: list[float],
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """
        Mix frequency-domain components of multiple images.

        Args:
            images: List of ImageModel instances (may contain None for empty slots).
            mode: Mixing mode — 'mag-phase' or 'real-imag'.
            weights_a: Per-image weights for component A (magnitude or real).
            weights_b: Per-image weights for component B (phase or imaginary).
            mask: 2D float64 binary mask (same shape as spectra), or None for all-ones.

        Returns:
            2D float64 spatial image normalized to 0.0–1.0.

        Raises:
            ValueError: If mode is not 'mag-phase' or 'real-imag'.
        """
        valid_modes = {"mag-phase", "real-imag"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {sorted(valid_modes)}"
            )

        # Collect loaded images and their spectra
        loaded_indices: list[int] = []
        spectra: list[np.ndarray] = []

        for i, img in enumerate(images):
            if img is not None and img.is_loaded:
                loaded_indices.append(i)
                spectra.append(img.ensure_fourier().spectrum)

        # No loaded images → zero array
        if not spectra:
            return np.zeros((256, 256), dtype=np.float64)

        shape = spectra[0].shape

        # Build mask
        if mask is None:
            mask = np.ones(shape, dtype=np.float64)

        # Build mixed spectrum
        if mode == "mag-phase":
            mixed_mag = np.zeros(shape, dtype=np.float64)
            mixed_phase = np.zeros(shape, dtype=np.float64)

            for idx, spectrum in zip(loaded_indices, spectra):
                wa = weights_a[idx]
                wb = weights_b[idx]
                mixed_mag += wa * np.abs(spectrum) * mask
                mixed_phase += wb * np.angle(spectrum) * mask

            # Polar → complex: mag * exp(j * phase)
            mixed_spectrum = mixed_mag * np.exp(1j * mixed_phase)

        else:
            # real-imag mode
            mixed_re = np.zeros(shape, dtype=np.float64)
            mixed_im = np.zeros(shape, dtype=np.float64)

            for idx, spectrum in zip(loaded_indices, spectra):
                wa = weights_a[idx]
                wb = weights_b[idx]
                mixed_re += wa * np.real(spectrum) * mask
                mixed_im += wb * np.imag(spectrum) * mask

            mixed_spectrum = mixed_re + 1j * mixed_im

        # IFFT: un-shift then inverse transform
        spatial = np.real(np.fft.ifft2(np.fft.ifftshift(mixed_spectrum)))

        # Normalize to 0.0–1.0
        s_min = spatial.min()
        s_max = spatial.max()
        if s_max > s_min:
            spatial = (spatial - s_min) / (s_max - s_min)
        else:
            spatial = np.zeros(shape, dtype=np.float64)

        return spatial
