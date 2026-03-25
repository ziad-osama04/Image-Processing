/**
 * API client for image processing backend.
 * Typed wrappers around fetch for all Phase 1 endpoints.
 */

import type {
  SessionResponse,
  ImageSlotResponse,
  FTComponentResponse,
  ResizePolicyRequest,
  ResizePolicyResponse,
  ReconstructResponse,
  VerifyRoundTripResponse,
  FTComponent,
} from '../types/image';

const BASE_URL = '/api/v1';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// ── Session ─────────────────────────────────────────────────────────

export async function createSession(): Promise<SessionResponse> {
  const res = await fetch(`${BASE_URL}/session`, { method: 'POST' });
  return handleResponse<SessionResponse>(res);
}

// ── Images ──────────────────────────────────────────────────────────

export async function uploadImage(
  sessionId: string,
  slot: number,
  file: File
): Promise<ImageSlotResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE_URL}/session/${sessionId}/images/${slot}`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<ImageSlotResponse>(res);
}

export async function getImage(
  sessionId: string,
  slot: number,
  brightness = 0,
  contrast = 1
): Promise<ImageSlotResponse> {
  const params = new URLSearchParams({
    brightness: brightness.toString(),
    contrast: contrast.toString(),
  });
  const res = await fetch(
    `${BASE_URL}/session/${sessionId}/images/${slot}?${params}`
  );
  return handleResponse<ImageSlotResponse>(res);
}

// ── FT Components ───────────────────────────────────────────────────

export async function getFTComponent(
  sessionId: string,
  slot: number,
  component: FTComponent,
  brightness = 0,
  contrast = 1
): Promise<FTComponentResponse> {
  const params = new URLSearchParams({
    brightness: brightness.toString(),
    contrast: contrast.toString(),
  });
  const res = await fetch(
    `${BASE_URL}/session/${sessionId}/images/${slot}/ft/${component}?${params}`
  );
  return handleResponse<FTComponentResponse>(res);
}

// ── Resize Policy ───────────────────────────────────────────────────

export async function setResizePolicy(
  sessionId: string,
  policy: ResizePolicyRequest
): Promise<ResizePolicyResponse> {
  const res = await fetch(`${BASE_URL}/session/${sessionId}/resize-policy`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(policy),
  });
  return handleResponse<ResizePolicyResponse>(res);
}

export async function getResizePolicy(
  sessionId: string
): Promise<ResizePolicyResponse> {
  const res = await fetch(`${BASE_URL}/session/${sessionId}/resize-policy`);
  return handleResponse<ResizePolicyResponse>(res);
}

// ── Reconstruction ──────────────────────────────────────────────────

export async function reconstructImage(
  sessionId: string,
  slot: number
): Promise<ReconstructResponse> {
  const res = await fetch(
    `${BASE_URL}/session/${sessionId}/images/${slot}/reconstruct`,
    { method: 'POST' }
  );
  return handleResponse<ReconstructResponse>(res);
}

export async function verifyRoundTrip(
  sessionId: string,
  slot: number
): Promise<VerifyRoundTripResponse> {
  const res = await fetch(
    `${BASE_URL}/session/${sessionId}/images/${slot}/verify-roundtrip`
  );
  return handleResponse<VerifyRoundTripResponse>(res);
}
