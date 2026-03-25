"""
ResizePolicy: Global resize policy for unifying image dimensions.

All resize math lives here — no resize logic in UI (Constitution §II).
"""

from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image


class ResizeMode(str, Enum):
    """Resize strategy for unifying image dimensions."""
    SMALLEST = "smallest"
    LARGEST = "largest"
    FIXED = "fixed"


class ResizePolicy:
    """
    Global policy for unifying image dimensions.

    Attributes:
        mode: Resize strategy (smallest, largest, or fixed).
        fixed_width: Target width for FIXED mode.
        fixed_height: Target height for FIXED mode.
        preserve_aspect: If True, pad/crop to preserve aspect ratio.
                         If False, force-stretch to target.
    """

    def __init__(
        self,
        mode: ResizeMode = ResizeMode.SMALLEST,
        fixed_width: Optional[int] = None,
        fixed_height: Optional[int] = None,
        preserve_aspect: bool = False,
    ):
        self.mode = mode
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.preserve_aspect = preserve_aspect

        if mode == ResizeMode.FIXED:
            if not fixed_width or not fixed_height:
                raise ValueError("FIXED mode requires both fixed_width and fixed_height")
            if fixed_width <= 0 or fixed_height <= 0:
                raise ValueError("Fixed dimensions must be positive")

    def compute_target_size(
        self, image_sizes: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """
        Compute the unified target size from a list of (height, width) tuples.

        Args:
            image_sizes: List of (height, width) for each loaded image.

        Returns:
            (target_height, target_width) tuple.
        """
        if not image_sizes:
            raise ValueError("No image sizes provided")

        if self.mode == ResizeMode.FIXED:
            return (self.fixed_height, self.fixed_width)  # type: ignore

        heights = [s[0] for s in image_sizes]
        widths = [s[1] for s in image_sizes]

        if self.mode == ResizeMode.SMALLEST:
            return (min(heights), min(widths))
        elif self.mode == ResizeMode.LARGEST:
            return (max(heights), max(widths))
        else:
            raise ValueError(f"Unknown resize mode: {self.mode}")

    @staticmethod
    def resize_image(
        array: np.ndarray,
        target_size: tuple[int, int],
        preserve_aspect: bool = False,
    ) -> np.ndarray:
        """
        Resize a 2D grayscale array to target_size (height, width).

        Args:
            array: Input 2D numpy array (grayscale, float64, 0-1).
            target_size: (target_height, target_width).
            preserve_aspect: If True, resize with aspect ratio preserved
                             and pad with zeros. If False, force-stretch.

        Returns:
            Resized 2D numpy array (float64, 0-1).
        """
        target_h, target_w = target_size
        src_h, src_w = array.shape[:2]

        if src_h == target_h and src_w == target_w:
            return array.copy()

        # Convert to PIL for resize
        img_uint8 = (array * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")

        if preserve_aspect:
            # Compute scale factor to fit within target while preserving ratio
            scale = min(target_w / src_w, target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Pad to target size (center the image)
            result = Image.new("L", (target_w, target_h), 0)
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            result.paste(resized, (paste_x, paste_y))
            return np.array(result, dtype=np.float64) / 255.0
        else:
            # Force-stretch to target
            resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            return np.array(resized, dtype=np.float64) / 255.0
