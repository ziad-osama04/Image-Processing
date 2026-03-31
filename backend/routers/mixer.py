"""
Mixer router: REST endpoint for FT component mixing.

Delegates all math to RegionMask and MixingEngine domain classes.
No business logic here — only orchestration (Constitution §V).
"""

import asyncio
import base64
import io

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from backend.domain.image_model import ImageModel
from backend.domain.mixing_engine import MixingEngine
from backend.domain.region_mask import RegionMask
from backend.services.image_store import image_store


router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────


class ImageWeight(BaseModel):
    """Weight pair for one image slot."""
    componentA: float
    componentB: float


class MixRequest(BaseModel):
    """Request body for the mix endpoint."""
    mode: str                           # 'mag-phase' | 'real-imag'
    weights: list[ImageWeight]          # exactly 4 entries
    region_size: float = 100.0          # 0–100 percentage
    region_type: str = "inner"          # 'inner' | 'outer'
    output_slot: int = 0                # 0 or 1
    simulate_slow: bool = False


class MixResponse(BaseModel):
    """Response body for the mix endpoint."""
    output_slot: int
    preview: str                        # base64 PNG
    width: int
    height: int


# ── Helpers ──────────────────────────────────────────────────────────


def _array_to_base64_png(array: np.ndarray) -> str:
    """Convert a 2D uint8 array to base64-encoded PNG string."""
    pil_img = PILImage.fromarray(array, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Mix endpoint ─────────────────────────────────────────────────────


@router.post(
    "/session/{session_id}/mix",
    response_model=MixResponse,
)
async def mix_images(session_id: str, request: MixRequest) -> MixResponse:
    """
    Mix FT components of loaded images using weighted combination.

    Applies a region mask and reconstructs a spatial image via IFFT.
    Stores the result in an output slot (4 or 5) of the session.
    """
    # Validate session
    session = image_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Validate mode
    if request.mode not in ("mag-phase", "real-imag"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Must be 'mag-phase' or 'real-imag'.",
        )

    # Validate output_slot
    if request.output_slot not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_slot {request.output_slot}. Must be 0 or 1.",
        )

    # Validate region_size
    if not (0.0 <= request.region_size <= 100.0):
        raise HTTPException(
            status_code=400,
            detail=f"region_size must be between 0 and 100, got {request.region_size}.",
        )

    # Validate region_type
    if request.region_type not in ("inner", "outer"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region_type '{request.region_type}'. Must be 'inner' or 'outer'.",
        )

    # Collect images from input slots 0–3
    images: list[ImageModel | None] = []
    first_loaded: ImageModel | None = None
    for slot_idx in range(session.MAX_INPUT_SLOTS):
        img = session.get_image(slot_idx)
        if img is not None and img.is_loaded:
            images.append(img)
            if first_loaded is None:
                first_loaded = img
        else:
            images.append(None)

    # At least 1 image must be loaded
    if first_loaded is None:
        raise HTTPException(
            status_code=422,
            detail="At least one image must be loaded to mix.",
        )

    # Build weights lists from request
    weights_a: list[float] = [w.componentA for w in request.weights]
    weights_b: list[float] = [w.componentB for w in request.weights]

    # Build region mask using shape of first loaded image
    active_array = first_loaded.get_active_array()
    height, width = active_array.shape[:2]
    region_mask = RegionMask(
        size_percent=request.region_size,
        mode=request.region_type,
    )
    mask = region_mask.to_array(height, width)

    # ── Normalize all images to the reference size before mixing ─────────
    target_size = (height, width)
    for img in images:
        if img is not None and img.is_loaded:
            arr = img.get_active_array()
            if arr.shape[:2] != target_size:
                img.apply_resize(target_size, preserve_aspect=False)

    # Simulate slow processing if requested (non-blocking)
    if request.simulate_slow:
        await asyncio.sleep(10)

    # Run CPU-bound mix in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(
        MixingEngine.mix, images, request.mode, weights_a, weights_b, mask
    )

    # Normalize to uint8
    result_uint8 = (result * 255).astype(np.uint8)

    # Convert to base64 PNG
    preview = _array_to_base64_png(result_uint8)

    # Store result as ImageModel in output slot
    output_model = ImageModel()
    output_model.filename = f"mix_output_{request.output_slot}.png"
    output_model.grayscale = result
    actual_slot = request.output_slot + session.MAX_INPUT_SLOTS  # 0→4, 1→5
    session.set_image(actual_slot, output_model)

    return MixResponse(
        output_slot=request.output_slot,
        preview=preview,
        width=width,
        height=height,
    )
