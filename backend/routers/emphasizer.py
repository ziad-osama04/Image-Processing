import asyncio
import base64
import io

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image as PILImage
from pydantic import BaseModel

from backend.domain.transform_engine import TransformEngine
from backend.services.image_store import image_store


router = APIRouter()


class TransformRequest(BaseModel):
    operation: str
    params: dict = {}
    domain: str = "spatial"
    slot: int = 0


class TransformResponse(BaseModel):
    request_id: int


class TransformResultResponse(BaseModel):
    request_id: int
    preview: str
    ft_preview: str
    width: int
    height: int


class TransformComponentResponse(BaseModel):
    image: str


class BottleneckRequest(BaseModel):
    enabled: bool


def _array_to_base64_png(array: np.ndarray) -> str:
    """Convert a 2D float array (0-1) to base64 PNG."""
    arr_clipped = np.clip(array, 0.0, 1.0)
    arr_uint8 = (arr_clipped * 255).astype(np.uint8)
    pil_img = PILImage.fromarray(arr_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _complex_to_ft_magnitude_base64(complex_array: np.ndarray) -> str:
    """Convert a 2D complex array to a log-magnitude base64 PNG (shifted)."""
    shifted = np.fft.fftshift(complex_array)
    mag = np.abs(shifted)
    log_mag = np.log1p(mag)
    if log_mag.max() > 0:
        log_mag /= log_mag.max()
    return _array_to_base64_png(log_mag)


def execute_transform(cancel_event, set_progress, operation, params, domain, input_array):
    """
    Background job to run the transform and compute outputs.
    """
    set_progress(0.1)
    if cancel_event.is_set(): return None

    if domain == "frequency":
        target = np.fft.fft2(input_array)
    else:
        target = input_array.astype(np.float64)

    set_progress(0.2)
    if cancel_event.is_set(): return None

    # Apply Transform
    try:
        if operation == "shift":
            res = TransformEngine.shift(target, params.get("shiftX", 0), params.get("shiftY", 0))
        elif operation == "complex-exponential":
            res = TransformEngine.complex_exponential(target, params.get("expU", 0.0), params.get("expV", 0.0))
        elif operation == "stretch":
            res = TransformEngine.stretch(target, params.get("stretchFactor", 1.0))
        elif operation == "mirror":
            res = TransformEngine.mirror(target, params.get("mirrorAxis", "horizontal"))
        elif operation == "even-odd":
            res = TransformEngine.even_odd(target, params.get("evenOddType", "even"))
        elif operation == "rotate":
            res = TransformEngine.rotate(target, params.get("rotateAngle", 0.0))
        elif operation == "differentiate":
            res = TransformEngine.differentiate(target, params.get("diffDirection", "both"))
        elif operation == "integrate":
            res = TransformEngine.integrate(target, params.get("intDirection", "both"))
        elif operation == "window":
            res = TransformEngine.window_2d(
                target, 
                params.get("windowType", "rectangular"),
                params.get("windowKernelWidth", 5),
                params.get("windowKernelHeight", 5),
                params.get("windowStrideX", 1),
                params.get("windowStrideY", 1),
                params.get("windowSigma", 1.0),
                params.get("windowMode", "same"),
            )
        else:
            res = target

        # Apply repeated FT ON TOP of the previous action!
        ft_count = params.get("ftCount", 0)
        if ft_count > 0:
            res = TransformEngine.repeated_ft(res, ft_count)

    except Exception as e:
        return {"error": str(e)}

    set_progress(0.8)
    if cancel_event.is_set(): return None

    # Compute Results — keep as complex!
    if domain == "frequency":
        mod_fft = res
        mod_spatial = np.fft.ifft2(mod_fft)  # Removed np.real to preserve phase!
    else:
        mod_spatial = res
        mod_fft = np.fft.fft2(mod_spatial)

    set_progress(0.9)
    if cancel_event.is_set(): return None

    # Compute default previews (magnitude)
    spatial_mag = np.abs(mod_spatial)
    s_min, s_max = spatial_mag.min(), spatial_mag.max()
    if s_max > s_min:
        spatial_normalized = (spatial_mag - s_min) / (s_max - s_min)
    else:
        spatial_normalized = spatial_mag

    preview = _array_to_base64_png(spatial_normalized)
    ft_preview = _complex_to_ft_magnitude_base64(mod_fft)
    h, w = mod_spatial.shape[:2]

    set_progress(1.0)
    return {
        "preview": preview,
        "ft_preview": ft_preview,
        "width": w,
        "height": h,
        "spatial_complex": mod_spatial,  # Expose raw arrays for dynamic component fetch
        "ft_complex": mod_fft
    }


@router.post("/session/{session_id}/transform", response_model=TransformResponse)
async def submit_transform(session_id: str, request: TransformRequest) -> TransformResponse:
    session = image_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    image = session.get_image(request.slot)
    if not image or not image.is_loaded:
        raise HTTPException(status_code=400, detail="Image not loaded")

    req_id = await session.job_manager.submit(
        execute_transform,
        request.operation,
        request.params,
        request.domain,
        image.get_active_array()
    )

    return TransformResponse(request_id=req_id)


@router.get("/session/{session_id}/progress")
async def stream_progress(session_id: str):
    session = image_store.get_session(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        jm = session.job_manager
        reported_id, progress = jm.get_progress()

        while True:
            current_id, current_progress = jm.get_progress()
            if current_id != reported_id and reported_id != 0: break
            reported_id = current_id
            
            yield f"data: {{\"request_id\": {current_id}, \"progress\": {current_progress}}}\n\n"
            
            if current_progress >= 1.0: break
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/session/{session_id}/transform-result/{request_id}", response_model=TransformResultResponse)
async def get_transform_result(session_id: str, request_id: int) -> TransformResultResponse:
    session = image_store.get_session(session_id)
    if not session or not session.job_manager.is_latest(request_id):
        raise HTTPException(status_code=409, detail="Request superseded")

    result = session.job_manager.get_result()
    if not result: raise HTTPException(status_code=400, detail="Result not ready")
    if "error" in result: raise HTTPException(status_code=500, detail=result["error"])

    return TransformResultResponse(
        request_id=request_id,
        preview=result["preview"],
        ft_preview=result["ft_preview"],
        width=result["width"],
        height=result["height"]
    )


@router.get(
    "/session/{session_id}/transform-result/{request_id}/{domain}/{component}",
    response_model=TransformComponentResponse,
)
async def get_transform_component(session_id: str, request_id: int, domain: str, component: str):
    """Dynamically extract mag/phase/real/imag from the stored complex transform result."""
    session = image_store.get_session(session_id)
    if not session or not session.job_manager.is_latest(request_id):
        raise HTTPException(status_code=409, detail="Request superseded")

    result = session.job_manager.get_result()
    if not result: raise HTTPException(status_code=400, detail="Result not ready")
    
    complex_arr = result["spatial_complex"] if domain == "spatial" else result["ft_complex"]
    arr = np.fft.fftshift(complex_arr) if domain == "frequency" else complex_arr
        
    if component == "magnitude":
        mag = np.abs(arr)
        val = np.log1p(mag) if domain == "frequency" else mag
        v_min, v_max = val.min(), val.max()
        norm = (val - v_min) / (v_max - v_min) if v_max > v_min else val
    elif component == "phase":
        val = np.angle(arr)
        norm = (val + np.pi) / (2 * np.pi)
    elif component == "real":
        val = np.real(arr)
        v_min, v_max = val.min(), val.max()
        norm = (val - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(val)
    elif component == "imaginary":
        val = np.imag(arr)
        v_min, v_max = val.min(), val.max()
        norm = (val - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(val)
    else:
        raise HTTPException(status_code=400, detail="Invalid component")
        
    return TransformComponentResponse(image=_array_to_base64_png(norm))


@router.post("/session/{session_id}/bottleneck")
async def toggle_bottleneck(session_id: str, request: BottleneckRequest):
    session = image_store.get_session(session_id)
    if session: session.job_manager.bottleneck = request.enabled
    return {"status": "success"}
