const BASE_URL = '/api/v1';

export interface TransformRequest {
  operation: string;
  params: Record<string, unknown>;
  domain: 'spatial' | 'frequency';
  slot: number;
}

export interface TransformResponse {
  request_id: number;
}

export interface TransformResultResponse {
  request_id: number;
  preview: string;       
  ft_preview: string;    
  width: number;
  height: number;
}

export interface TransformComponentResponse {
  image: string; // Base64 PNG
}

export interface ProgressEvent {
  request_id: number;
  progress: number;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function applyTransform(sessionId: string, request: TransformRequest, signal?: AbortSignal): Promise<TransformResponse> {
  const res = await fetch(`${BASE_URL}/session/${sessionId}/transform`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  return handleResponse<TransformResponse>(res);
}

export function connectProgress(
  sessionId: string,
  onProgress: (event: ProgressEvent) => void,
  onComplete: () => void,
  onError?: (error: Event) => void,
): () => void {
  const url = `${BASE_URL}/session/${sessionId}/progress`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data: ProgressEvent = JSON.parse(event.data);
      onProgress(data);
      if (data.progress >= 1.0) {
        eventSource.close();
        onComplete();
      }
    } catch { /* Ignore */ }
  };

  eventSource.onerror = (event) => {
    eventSource.close();
    onError?.(event);
  };
  return () => eventSource.close();
}

export async function getTransformResult(sessionId: string, requestId: number, signal?: AbortSignal): Promise<TransformResultResponse> {
  const res = await fetch(`${BASE_URL}/session/${sessionId}/transform-result/${requestId}`, { signal });
  return handleResponse<TransformResultResponse>(res);
}

export async function getTransformComponent(sessionId: string, requestId: number, domain: 'spatial' | 'frequency', component: string, signal?: AbortSignal): Promise<TransformComponentResponse> {
  const res = await fetch(`${BASE_URL}/session/${sessionId}/transform-result/${requestId}/${domain}/${component}`, { signal });
  return handleResponse<TransformComponentResponse>(res);
}

export async function toggleBottleneck(sessionId: string, enabled: boolean): Promise<void> {
  await fetch(`${BASE_URL}/session/${sessionId}/bottleneck`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
}
