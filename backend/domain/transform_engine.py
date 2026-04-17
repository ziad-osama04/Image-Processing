"""
TransformEngine: All 10 FT property operations.

Pure-function class implementing shift, complex exponential,
stretch, mirror, even/odd, rotation, differentiation, integration,
2D windowing, and repeated FT.

All methods are static — no instance state (Constitution §II).
Uses NumPy vectorized operations for performance.
Optimized: separable convolution, scipy.fft for speed.
"""

import numpy as np
from scipy import ndimage, signal
from scipy.fft import fft2 as scipy_fft2


class TransformEngine:
    """
    Implements all 10 FT property operations as static methods.

    Every method takes a numpy array (2D, float64 or complex128)
    and returns a transformed numpy array. No side effects.
    """

    # ── 1. Shift ─────────────────────────────────────────────────────

    @staticmethod
    def shift(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
        return np.roll(array, (dy, dx), axis=(0, 1))

    # ── 2. Complex Exponential ───────────────────────────────────────

    @staticmethod
    def complex_exponential(
        array: np.ndarray, fx: float, fy: float
    ) -> np.ndarray:
        h, w = array.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        phase = 2.0 * np.pi * (fx * x_coords / w + fy * y_coords / h)
        exponential = np.exp(1j * phase)
        return array * exponential

    # ── 3. Stretch ───────────────────────────────────────────────────

    @staticmethod
    def stretch(array: np.ndarray, factor: float) -> np.ndarray:
        if factor <= 0:
            raise ValueError(f"Stretch factor must be > 0, got {factor}")
        return ndimage.zoom(array, factor, order=3)

    # ── 4. Mirror ────────────────────────────────────────────────────

    @staticmethod
    def mirror(array: np.ndarray, axis: str) -> np.ndarray:
        """
        Create symmetry by duplicating half the array onto the other half
        (kaleidoscope-style), as required by the spec.

        Idempotent: once the image is symmetric, applying again leaves it
        unchanged — f(f(x)) == f(x).
        """
        if axis not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Invalid axis '{axis}'")

        h, w = array.shape[:2]
        out = array.copy()

        if axis in ("horizontal", "both"):
            mid = w // 2
            # Mirror the left half onto the right half
            out[:, mid:] = np.flip(out[:, : w - mid], axis=1)

        if axis in ("vertical", "both"):
            mid = h // 2
            # Mirror the top half onto the bottom half
            # (reads from `out` so horizontal mirror is respected for "both")
            out[mid:, :] = np.flip(out[: h - mid, :], axis=0)

        return out

    # ── 5. Even/Odd ──────────────────────────────────────────────────

    @staticmethod
    def even_odd(array: np.ndarray, mode: str) -> np.ndarray:
        flipped = np.flip(array, axis=(0, 1))
        if mode == "even":
            return (array + flipped) / 2.0
        elif mode == "odd":
            return (array - flipped) / 2.0
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'even' or 'odd'.")

    # ── 6. Rotate ────────────────────────────────────────────────────

    @staticmethod
    def rotate(array: np.ndarray, angle_deg: float) -> np.ndarray:
        return ndimage.rotate(array, angle_deg, reshape=True, order=3)

    # ── 7. Differentiate ─────────────────────────────────────────────

    @staticmethod
    def differentiate(array: np.ndarray, direction: str = "both") -> np.ndarray:
        gy, gx = np.gradient(array.astype(np.complex128))
        if direction == "x":
            return gx
        elif direction == "y":
            return gy
        else:
            # Magnitude of complex gradient
            return np.sqrt(np.abs(gx)**2 + np.abs(gy)**2)

    # ── 8. Integrate ─────────────────────────────────────────────────

    @staticmethod
    def integrate(array: np.ndarray, direction: str = "both") -> np.ndarray:
        if direction == "x":
            return np.cumsum(array.astype(np.complex128), axis=1)
        elif direction == "y":
            return np.cumsum(array.astype(np.complex128), axis=0)
        else:
            # 2D cumulative sum
            return np.cumsum(np.cumsum(array.astype(np.complex128), axis=0), axis=1)

    # ── 9. 2D Windowing ──────────────────────────────────────────────

    @staticmethod
    def window_2d(
        array: np.ndarray,
        window_type: str,
        kernel_width: int = 5,
        kernel_height: int = 5,
        stride_x: int = 1,
        stride_y: int = 1,
        sigma: float = 1.0,
        mode: str = "same",
    ) -> np.ndarray:
        """Apply a sliding-window convolution using the selected window kernel.

        Uses separable 1D convolution passes for O(n*(kw+kh)) instead of
        O(n*kw*kh). Builds two 1D windows, normalizes so their product
        sums to 1, then applies horizontal and vertical ndimage.convolve1d.

        Parameters
        ----------
        array : 2D ndarray (float64 or complex128)
        window_type : rectangular | gaussian | hamming | hanning
        kernel_width, kernel_height : odd int, size of the convolution kernel
        stride_x, stride_y : step size of the sliding window
        sigma : std-dev ratio for gaussian window
        mode : 'same' (output = input size) or 'valid' (no padding)
        """
        # Ensure odd kernel sizes
        kw = max(1, kernel_width) | 1
        kh = max(1, kernel_height) | 1

        # ── Build 1D windows ───────────────────────────────────────────
        if window_type == "rectangular":
            win_h = np.ones(kh)
            win_w = np.ones(kw)
        elif window_type == "gaussian":
            win_h = signal.windows.gaussian(kh, std=max(0.1, sigma * kh / 4))
            win_w = signal.windows.gaussian(kw, std=max(0.1, sigma * kw / 4))
        elif window_type == "hamming":
            win_h = signal.windows.hamming(kh)
            win_w = signal.windows.hamming(kw)
        elif window_type == "hanning":
            win_h = signal.windows.hann(kh)
            win_w = signal.windows.hann(kw)
        else:
            raise ValueError(f"Unknown window_type '{window_type}'.")

        # ── Normalize so product sums to 1 ─────────────────────────────
        total_sum = win_h.sum() * win_w.sum()
        if total_sum > 0:
            # Normalize one of them so the separable product sums to 1
            win_h = win_h / win_h.sum()
            win_w = win_w / win_w.sum()

        # ── Separable convolution (two 1D passes) ──────────────────────
        ndimage_mode = "constant" if mode == "same" else "constant"
        origin_h = 0
        origin_w = 0

        if np.iscomplexobj(array):
            # Horizontal pass
            tmp_real = ndimage.convolve1d(array.real, win_w, axis=1, mode=ndimage_mode, cval=0.0, origin=origin_w)
            tmp_imag = ndimage.convolve1d(array.imag, win_w, axis=1, mode=ndimage_mode, cval=0.0, origin=origin_w)
            # Vertical pass
            result_real = ndimage.convolve1d(tmp_real, win_h, axis=0, mode=ndimage_mode, cval=0.0, origin=origin_h)
            result_imag = ndimage.convolve1d(tmp_imag, win_h, axis=0, mode=ndimage_mode, cval=0.0, origin=origin_h)
            result = result_real + 1j * result_imag
        else:
            tmp = ndimage.convolve1d(array, win_w, axis=1, mode=ndimage_mode, cval=0.0, origin=origin_w)
            result = ndimage.convolve1d(tmp, win_h, axis=0, mode=ndimage_mode, cval=0.0, origin=origin_h)

        # Crop for 'valid' mode
        if mode == "valid":
            ph = kh // 2
            pw = kw // 2
            result = result[ph:result.shape[0]-ph, pw:result.shape[1]-pw]

        # ── Apply stride (subsample) ──────────────────────────────────
        sx = max(1, int(stride_x))
        sy = max(1, int(stride_y))
        if sx > 1 or sy > 1:
            result = result[::sy, ::sx]

        return result

    # ── 10. Repeated FT ──────────────────────────────────────────────

    @staticmethod
    def repeated_ft(array: np.ndarray, count: int) -> np.ndarray:
        if count < 1:
            return array.copy()

        result = array.astype(np.complex128)
        for _ in range(count):
            result = scipy_fft2(result, workers=-1)
        return result