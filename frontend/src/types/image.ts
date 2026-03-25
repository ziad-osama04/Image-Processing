/**
 * TypeScript interfaces for the image processing API.
 */

export interface SessionResponse {
  session_id: string;
}

export interface ImageSlotResponse {
  slot: number;
  filename: string;
  width: number;
  height: number;
  preview: string; // base64 PNG
}

export interface FTComponentResponse {
  slot: number;
  component: string;
  image: string; // base64 PNG
}

export interface ResizePolicyRequest {
  mode: 'smallest' | 'largest' | 'fixed';
  fixed_width?: number;
  fixed_height?: number;
  preserve_aspect: boolean;
}

export interface ResizePolicyResponse {
  mode: string;
  fixed_width?: number;
  fixed_height?: number;
  preserve_aspect: boolean;
  target_width?: number;
  target_height?: number;
  affected_slots: number[];
}

export interface ReconstructResponse {
  slot: number;
  image: string; // base64 PNG
}

export interface VerifyRoundTripResponse {
  slot: number;
  max_error: number;
  passed: boolean;
}

export type FTComponent = 'magnitude' | 'phase' | 'real' | 'imaginary';

export interface ViewportState {
  sessionId: string;
  slot: number;
  activeComponent: 'spatial' | FTComponent;
  brightness: number;
  contrast: number;
  imageSrc: string | null;
  filename: string;
  width: number;
  height: number;
}
