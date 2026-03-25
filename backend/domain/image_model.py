"""
ImageModel: Domain entity for a single loaded image.

Owns original data, grayscale conversion, resize state,
FT access, and display-only B/C adjustment.
All image math lives here (Constitution §II).
"""

from typing import Optional

import numpy as np
from PIL import Image as PILImage

from backend.domain.fourier_data import FourierData


# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp"}


class ImageModel:
    """
    Represents a single loaded image with all derived state.

    Attributes:
        filename: Original filename.
        original_data: Raw pixel data as loaded (H×W×C uint8 or H×W uint8).
        grayscale: Normalized grayscale (H×W float64, 0.0-1.0).
        resized: Grayscale after resize policy (or None).
        fourier: Cached FourierData (or None, lazy-computed).
    """

    def __init__(self):
        self.filename: str = ""
        self.original_data: Optional[np.ndarray] = None
        self.grayscale: Optional[np.ndarray] = None
        self.resized: Optional[np.ndarray] = None
        self.fourier: Optional[FourierData] = None

    @property
    def is_loaded(self) -> bool:
        """Whether an image has been loaded."""
        return self.grayscale is not None

    def load(self, file_bytes: bytes, filename: str) -> None:
        """
        Load an image from bytes.

        Converts to grayscale and normalizes to 0.0-1.0 float64.
        Clears resize and FT cache.

        Args:
            file_bytes: Raw file bytes (PNG/JPG/BMP).
            filename: Original filename for format validation.

        Raises:
            ValueError: If the file format is not supported.
        """
        import io
        from pathlib import Path

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        pil_img = PILImage.open(io.BytesIO(file_bytes))
        self.original_data = np.array(pil_img)
        self.filename = filename

        # Convert to grayscale
        if pil_img.mode in ("RGB", "RGBA"):
            gray_img = pil_img.convert("L")
        elif pil_img.mode == "L":
            gray_img = pil_img
        else:
            gray_img = pil_img.convert("L")

        # Normalize to 0.0-1.0 float64
        self.grayscale = np.array(gray_img, dtype=np.float64) / 255.0

        # Clear derived state
        self.resized = None
        self.fourier = None

    def apply_resize(self, target_size: tuple[int, int], preserve_aspect: bool = False) -> None:
        """
        Resize the grayscale image to target dimensions.

        Invalidates FT cache.

        Args:
            target_size: (target_height, target_width).
            preserve_aspect: Whether to preserve aspect ratio.
        """
        if self.grayscale is None:
            raise RuntimeError("No image loaded")

        from backend.domain.resize_policy import ResizePolicy

        self.resized = ResizePolicy.resize_image(
            self.grayscale, target_size, preserve_aspect
        )
        # Invalidate FT cache — must recompute
        self.fourier = None

    def get_active_array(self) -> np.ndarray:
        """
        Get the active image array (resized if available, else grayscale).

        Returns:
            2D float64 array (0.0-1.0).
        """
        if self.resized is not None:
            return self.resized
        if self.grayscale is not None:
            return self.grayscale
        raise RuntimeError("No image loaded")

    def ensure_fourier(self) -> FourierData:
        """
        Lazily compute FT if not cached.

        Returns:
            FourierData instance.
        """
        if self.fourier is None:
            self.fourier = FourierData.from_image(self.get_active_array())
        return self.fourier

    def get_display_image(
        self, brightness: float = 0.0, contrast: float = 1.0
    ) -> np.ndarray:
        """
        Get the display image with B/C applied.

        B/C is display-only — original data is never mutated (FR-012).
        FT computation always uses original/resized data, never B/C-adjusted.

        Args:
            brightness: Offset in [-1.0, 1.0], mapped to [-255, 255].
            contrast: Multiplier in [0.1, 10.0].

        Returns:
            2D uint8 array (0-255) with B/C applied.
        """
        active = self.get_active_array()
        img_uint8 = (active * 255).astype(np.float64)

        brightness = max(-1.0, min(1.0, brightness))
        contrast = max(0.1, min(10.0, contrast))

        adjusted = contrast * img_uint8 + brightness * 255
        return np.clip(adjusted, 0, 255).astype(np.uint8)
