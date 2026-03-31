"""
RegionMask: Centered rectangle frequency-domain mask.

Selects which frequencies participate in mixing.
'inner' = low-pass (center), 'outer' = high-pass (edges).
All mask math lives here — no duplication (Constitution §II).
"""

import numpy as np


class RegionMask:
    """
    A binary frequency-domain mask shaped as a centered rectangle.

    The mask is applied element-wise to FFT spectra (fftshifted, DC at center)
    before mixing, to select which frequency components participate.

    Attributes:
        size_percent: Rectangle size as percentage of spectrum dimensions (0–100).
        mode: 'inner' (low-pass) or 'outer' (high-pass).
    """

    VALID_MODES = {"inner", "outer"}

    def __init__(self, size_percent: float, mode: str) -> None:
        """
        Initialize a RegionMask.

        Args:
            size_percent: Rectangle size as percentage (0.0–100.0).
            mode: 'inner' for low-pass, 'outer' for high-pass.

        Raises:
            ValueError: If mode is not 'inner' or 'outer'.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {sorted(self.VALID_MODES)}"
            )
        self.size_percent = float(size_percent)
        self.mode = mode

    def to_array(self, height: int, width: int) -> np.ndarray:
        """
        Generate the 2D binary mask array.

        Centered rectangle with DC at center (fftshift convention).
        Matches fftEngine.ts createRegionMask() behavior exactly.

        Args:
            height: Number of rows in the spectrum.
            width: Number of columns in the spectrum.

        Returns:
            2D float64 array of shape (height, width) with values 0.0 or 1.0.
        """
        fraction = self.size_percent / 100.0
        half_h = (height * fraction) / 2.0
        half_w = (width * fraction) / 2.0
        center_r = height / 2.0
        center_c = width / 2.0

        # Build coordinate grids
        rows = np.arange(height)
        cols = np.arange(width)
        r_grid, c_grid = np.meshgrid(rows, cols, indexing="ij")

        # Determine which pixels fall inside the centered rectangle
        in_rect = (np.abs(r_grid - center_r) <= half_h) & (
            np.abs(c_grid - center_c) <= half_w
        )

        if self.mode == "inner":
            mask = np.where(in_rect, 1.0, 0.0)
        else:
            mask = np.where(in_rect, 0.0, 1.0)

        return mask
