"""
TransformEngine: All 10 FT property operations.

Pure-function class implementing shift, complex exponential,
stretch, mirror, even/odd, rotation, differentiation, integration,
2D windowing, and repeated FT.

All methods are static — no instance state (Constitution §II).
Uses NumPy vectorized operations for performance.
"""

import numpy as np
from scipy import ndimage, signal


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
        sigma: float = 1.0,
        width_ratio: float = 1.0,
        height_ratio: float = 1.0,
    ) -> np.ndarray:
        """Apply a resizable 2D window function."""
        h, w = array.shape[:2]

        win_w_len = max(1, int(w * width_ratio))
        win_h_len = max(1, int(h * height_ratio))

        if window_type == "rectangular":
            win_h_small = np.ones(win_h_len)
            win_w_small = np.ones(win_w_len)
        elif window_type == "gaussian":
            win_h_small = signal.windows.gaussian(win_h_len, std=sigma * win_h_len / 4)
            win_w_small = signal.windows.gaussian(win_w_len, std=sigma * win_w_len / 4)
        elif window_type == "hamming":
            win_h_small = signal.windows.hamming(win_h_len)
            win_w_small = signal.windows.hamming(win_w_len)
        elif window_type == "hanning":
            win_h_small = signal.windows.hann(win_h_len)
            win_w_small = signal.windows.hann(win_w_len)
        else:
            raise ValueError(f"Unknown window_type '{window_type}'.")

        # Zero-pad the smaller window to the full array size (centered)
        pad_left = (w - win_w_len) // 2
        pad_right = w - win_w_len - pad_left
        win_w = np.pad(win_w_small, (pad_left, pad_right), mode="constant")

        pad_top = (h - win_h_len) // 2
        pad_bottom = h - win_h_len - pad_top
        win_h = np.pad(win_h_small, (pad_top, pad_bottom), mode="constant")

        window_2d = np.outer(win_h, win_w)
        return array * window_2d

    # ── 10. Repeated FT ──────────────────────────────────────────────

    @staticmethod
    def repeated_ft(array: np.ndarray, count: int) -> np.ndarray:
        if count < 1:
            return array.copy()

        result = array.astype(np.complex128)
        for _ in range(count):
            result = np.fft.fft2(result)
        return result