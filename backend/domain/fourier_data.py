"""
FourierData: Frequency-domain decomposition of an image.

All FFT math lives here — no FFT logic in UI or routers (Constitution §II).
Cached component properties avoid redundant computation (Constitution §IX).
"""

import numpy as np


class FourierData:
    """
    Immutable frequency-domain decomposition of a 2D grayscale image.

    Attributes:
        spectrum: Complex128 FFT result (shifted, DC at center).
    """

    def __init__(self, spectrum: np.ndarray):
        """
        Initialize with a pre-computed FFT spectrum.

        Use `from_image()` class method for normal construction.
        """
        self._spectrum = spectrum
        self._magnitude: np.ndarray | None = None
        self._phase: np.ndarray | None = None
        self._real: np.ndarray | None = None
        self._imaginary: np.ndarray | None = None

    @classmethod
    def from_image(cls, array: np.ndarray) -> "FourierData":
        """
        Compute 2D FFT of a grayscale image.

        Args:
            array: 2D numpy array (float64, 0-1 normalized).

        Returns:
            FourierData instance with cached spectrum.
        """
        spectrum = np.fft.fftshift(np.fft.fft2(array))
        return cls(spectrum)

    @property
    def spectrum(self) -> np.ndarray:
        """Raw complex spectrum (shifted)."""
        return self._spectrum

    @property
    def magnitude(self) -> np.ndarray:
        """Log-scaled magnitude spectrum, normalized to 0-255 uint8."""
        if self._magnitude is None:
            mag = np.abs(self._spectrum)
            # Log scale: 20*log10(1 + |spectrum|)
            log_mag = 20 * np.log10(1 + mag)
            # Normalize to 0-255
            if log_mag.max() > 0:
                self._magnitude = (
                    (log_mag / log_mag.max()) * 255
                ).astype(np.uint8)
            else:
                self._magnitude = np.zeros_like(log_mag, dtype=np.uint8)
        return self._magnitude

    @property
    def phase(self) -> np.ndarray:
        """Phase spectrum, normalized to 0-255 uint8."""
        if self._phase is None:
            ph = np.angle(self._spectrum)
            # Normalize from [-pi, pi] to [0, 255]
            self._phase = (
                ((ph + np.pi) / (2 * np.pi)) * 255
            ).astype(np.uint8)
        return self._phase

    @property
    def real_part(self) -> np.ndarray:
        """Real part of spectrum, normalized to 0-255 uint8."""
        if self._real is None:
            real = np.real(self._spectrum)
            # Normalize to 0-255
            r_min, r_max = real.min(), real.max()
            if r_max > r_min:
                self._real = (
                    ((real - r_min) / (r_max - r_min)) * 255
                ).astype(np.uint8)
            else:
                self._real = np.zeros_like(real, dtype=np.uint8)
        return self._real

    @property
    def imaginary_part(self) -> np.ndarray:
        """Imaginary part of spectrum, normalized to 0-255 uint8."""
        if self._imaginary is None:
            imag = np.imag(self._spectrum)
            i_min, i_max = imag.min(), imag.max()
            if i_max > i_min:
                self._imaginary = (
                    ((imag - i_min) / (i_max - i_min)) * 255
                ).astype(np.uint8)
            else:
                self._imaginary = np.zeros_like(imag, dtype=np.uint8)
        return self._imaginary

    def get_component(self, name: str) -> np.ndarray:
        """
        Get a named FT component as uint8 array.

        Args:
            name: One of 'magnitude', 'phase', 'real', 'imaginary'.

        Returns:
            2D uint8 array (0-255).

        Raises:
            ValueError: If name is not a valid component.
        """
        components = {
            "magnitude": self.magnitude,
            "phase": self.phase,
            "real": self.real_part,
            "imaginary": self.imaginary_part,
        }
        if name not in components:
            raise ValueError(
                f"Invalid component '{name}'. "
                f"Must be one of: {list(components.keys())}"
            )
        return components[name]

    def get_display_component(
        self, name: str, brightness: float = 0.0, contrast: float = 1.0
    ) -> np.ndarray:
        """
        Get a named FT component with brightness/contrast applied.

        B/C is display-only — original cached data is never mutated (FR-012).

        Args:
            name: Component name.
            brightness: Offset in [-1.0, 1.0], mapped to [-255, 255].
            contrast: Multiplier in [0.1, 10.0].

        Returns:
            2D uint8 array with B/C applied.
        """
        component = self.get_component(name).astype(np.float64)
        brightness = max(-1.0, min(1.0, brightness))
        contrast = max(0.1, min(10.0, contrast))
        adjusted = contrast * component + brightness * 255
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct spatial image from spectrum via inverse FFT.

        Returns:
            2D float64 array (real part of IFFT result).
        """
        return np.real(np.fft.ifft2(np.fft.ifftshift(self._spectrum)))
