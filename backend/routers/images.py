"""
Images router: REST endpoints for session management, image operations,
FT components, resize policy, and reconstruction.

Delegates all logic to domain classes (Constitution §II).
"""

import base64
import io
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image as PILImage
from pydantic import BaseModel

from backend.domain.image_model import ImageModel
from backend.domain.resize_policy import ResizeMode, ResizePolicy
from backend.services.image_store import image_store

router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────


class SessionResponse(BaseModel):
    session_id: str


class ImageSlotResponse(BaseModel):
    slot: int
    filename: str
    width: int
    height: int
    preview: str  # base64 PNG


class FTComponentResponse(BaseModel):
    slot: int
    component: str
    image: str  # base64 PNG


class ResizePolicyRequest(BaseModel):
    mode: ResizeMode
    fixed_width: Optional[int] = None
    fixed_height: Optional[int] = None
    preserve_aspect: bool = False


class ResizePolicyResponse(BaseModel):
    mode: str
    fixed_width: Optional[int] = None
    fixed_height: Optional[int] = None
    preserve_aspect: bool
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    affected_slots: list[int] = []


class ReconstructResponse(BaseModel):
    slot: int
    image: str  # base64 PNG


class VerifyRoundTripResponse(BaseModel):
    slot: int
    max_error: float
    passed: bool


class ErrorResponse(BaseModel):
    detail: str


# ── Helpers ──────────────────────────────────────────────────────────


def _array_to_base64_png(array: np.ndarray) -> str:
    """Convert a 2D uint8 array to base64-encoded PNG string."""
    pil_img = PILImage.fromarray(array, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_session(session_id: str):
    """Get session or raise 404."""
    session = image_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _get_loaded_image(session, slot: int) -> ImageModel:
    """Get loaded image from session or raise 404."""
    if slot < 0 or slot >= session.TOTAL_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=f"Slot {slot} out of range (0-{session.TOTAL_SLOTS - 1})",
        )
    img = session.get_image(slot)
    if img is None or not img.is_loaded:
        raise HTTPException(status_code=404, detail=f"Slot {slot} is empty")
    return img


# ── Session endpoints ────────────────────────────────────────────────


@router.post("/session", response_model=SessionResponse)
async def create_session():
    """Create a new session."""
    session = image_store.create_session()
    return SessionResponse(session_id=session.id)


# ── Image endpoints (US1) ───────────────────────────────────────────


@router.post(
    "/session/{session_id}/images/{slot}",
    response_model=ImageSlotResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_image(session_id: str, slot: int, file: UploadFile = File(...)):
    """Upload or replace an image in a slot."""
    session = _get_session(session_id)

    if slot < 0 or slot >= session.MAX_INPUT_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=f"Input slot {slot} out of range (0-{session.MAX_INPUT_SLOTS - 1})",
        )

    file_bytes = await file.read()
    filename = file.filename or "unknown.png"

    img = ImageModel()
    try:
        img.load(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session.set_image(slot, img)

    # Auto-apply resize policy if active
    if session.resize_policy is not None:
        sizes = session.get_loaded_input_sizes()
        if len(sizes) >= 2 or session.resize_policy.mode == ResizeMode.FIXED:
            target = session.resize_policy.compute_target_size(sizes)
            img.apply_resize(target, session.resize_policy.preserve_aspect)

    active = img.get_active_array()
    preview = _array_to_base64_png(img.get_display_image())

    return ImageSlotResponse(
        slot=slot,
        filename=filename,
        width=active.shape[1],
        height=active.shape[0],
        preview=preview,
    )


@router.get(
    "/session/{session_id}/images/{slot}",
    response_model=ImageSlotResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_image(
    session_id: str,
    slot: int,
    brightness: float = Query(0.0, ge=-1.0, le=1.0),
    contrast: float = Query(1.0, ge=0.1, le=10.0),
):
    """Get display image for a slot with optional B/C adjustment."""
    session = _get_session(session_id)
    img = _get_loaded_image(session, slot)

    active = img.get_active_array()
    display = img.get_display_image(brightness, contrast)
    preview = _array_to_base64_png(display)

    return ImageSlotResponse(
        slot=slot,
        filename=img.filename,
        width=active.shape[1],
        height=active.shape[0],
        preview=preview,
    )


# ── FT Component endpoints (US2) ────────────────────────────────────


VALID_COMPONENTS = {"magnitude", "phase", "real", "imaginary"}


@router.get(
    "/session/{session_id}/images/{slot}/ft/{component}",
    response_model=FTComponentResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_ft_component(
    session_id: str,
    slot: int,
    component: str,
    brightness: float = Query(0.0, ge=-1.0, le=1.0),
    contrast: float = Query(1.0, ge=0.1, le=10.0),
):
    """Get a specific FT component for a slot (with B/C)."""
    if component not in VALID_COMPONENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component '{component}'. Must be one of: {sorted(VALID_COMPONENTS)}",
        )

    session = _get_session(session_id)
    img = _get_loaded_image(session, slot)

    fourier = img.ensure_fourier()
    display = fourier.get_display_component(component, brightness, contrast)
    image_b64 = _array_to_base64_png(display)

    return FTComponentResponse(slot=slot, component=component, image=image_b64)


# ── Resize Policy endpoints (US3) ───────────────────────────────────


@router.put(
    "/session/{session_id}/resize-policy",
    response_model=ResizePolicyResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def set_resize_policy(session_id: str, request: ResizePolicyRequest):
    """Set the global resize policy for a session."""
    session = _get_session(session_id)

    try:
        policy = ResizePolicy(
            mode=request.mode,
            fixed_width=request.fixed_width,
            fixed_height=request.fixed_height,
            preserve_aspect=request.preserve_aspect,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    affected = session.apply_resize_policy(policy)

    # Compute target for response
    sizes = session.get_loaded_input_sizes()
    target_h, target_w = None, None
    if sizes:
        try:
            target_h, target_w = policy.compute_target_size(sizes)
        except ValueError:
            pass

    return ResizePolicyResponse(
        mode=policy.mode.value,
        fixed_width=policy.fixed_width,
        fixed_height=policy.fixed_height,
        preserve_aspect=policy.preserve_aspect,
        target_width=target_w,
        target_height=target_h,
        affected_slots=affected,
    )


@router.get(
    "/session/{session_id}/resize-policy",
    response_model=ResizePolicyResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_resize_policy(session_id: str):
    """Get the current resize policy for a session."""
    session = _get_session(session_id)
    policy = session.resize_policy

    if policy is None:
        return ResizePolicyResponse(
            mode="smallest",
            preserve_aspect=False,
        )

    sizes = session.get_loaded_input_sizes()
    target_h, target_w = None, None
    if sizes:
        try:
            target_h, target_w = policy.compute_target_size(sizes)
        except ValueError:
            pass

    return ResizePolicyResponse(
        mode=policy.mode.value,
        fixed_width=policy.fixed_width,
        fixed_height=policy.fixed_height,
        preserve_aspect=policy.preserve_aspect,
        target_width=target_w,
        target_height=target_h,
    )


# ── Reconstruction endpoints (US6) ──────────────────────────────────


@router.post(
    "/session/{session_id}/images/{slot}/reconstruct",
    response_model=ReconstructResponse,
    responses={404: {"model": ErrorResponse}},
)
async def reconstruct_image(session_id: str, slot: int):
    """Reconstruct spatial image from FT via IFFT."""
    session = _get_session(session_id)
    img = _get_loaded_image(session, slot)

    fourier = img.ensure_fourier()
    reconstructed = fourier.reconstruct()

    # Normalize to uint8 for display
    r_min, r_max = reconstructed.min(), reconstructed.max()
    if r_max > r_min:
        display = ((reconstructed - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        display = np.zeros_like(reconstructed, dtype=np.uint8)

    return ReconstructResponse(
        slot=slot,
        image=_array_to_base64_png(display),
    )


@router.get(
    "/session/{session_id}/images/{slot}/verify-roundtrip",
    response_model=VerifyRoundTripResponse,
    responses={404: {"model": ErrorResponse}},
)
async def verify_roundtrip(session_id: str, slot: int):
    """Verify FT→IFFT round-trip fidelity (should be ≤ 1e-6)."""
    session = _get_session(session_id)
    img = _get_loaded_image(session, slot)

    original = img.get_active_array()
    fourier = img.ensure_fourier()
    reconstructed = fourier.reconstruct()

    max_error = float(np.max(np.abs(original - reconstructed)))

    return VerifyRoundTripResponse(
        slot=slot,
        max_error=max_error,
        passed=max_error <= 1e-6,
    )
