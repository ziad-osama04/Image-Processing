export interface SessionResponse {
  session_id: string;
}

export interface ImageSlotResponse {
  slot: number;
  filename: string;
  width: number;
  height: number;
  preview: string;
}

export interface FTComponentResponse {
  slot: number;
  component: string;
  image: string;
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
  image: string;
}

export interface VerifyRoundTripResponse {
  slot: number;
  max_error: number;
  passed: boolean;
}

export type FTComponent = 'magnitude' | 'phase' | 'real' | 'imaginary';

export interface ViewportPairProps {
  sessionId: string;
  slot: number;
  isInput: boolean;
  onImageLoaded?: (slot: number) => void;
}

export type OutputTarget = 0 | 1;

export interface ResizePolicyState {
  mode: 'smallest' | 'largest' | 'fixed';
  fixedWidth: number;
  fixedHeight: number;
  preserveAspect: boolean;
}

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

export type MixMode = 'mag-phase' | 'real-imag';

export interface ImageWeight {
  componentA: number;
  componentB: number;
}

export interface MixerState {
  mode: MixMode;
  weights: ImageWeight[];
  regionSize: number;
  regionType: 'inner' | 'outer';
  simulateSlow: boolean;
}

export interface MixRequest {
  mode: MixMode;
  weights: ImageWeight[];
  region_size: number;
  region_type: 'inner' | 'outer';
  output_slot: number;
  simulate_slow: boolean;
}

export interface MixResponse {
  output_slot: number;
  preview: string;
  width: number;
  height: number;
}

// NOTE: Removed 'multiple-ft' as a mutually exclusive action!
export type EmphasizerAction =
  | 'shift'
  | 'complex-exponential'
  | 'stretch'
  | 'mirror'
  | 'even-odd'
  | 'rotate'
  | 'differentiate'
  | 'integrate'
  | 'window';

export type WindowType = 'rectangular' | 'gaussian' | 'hamming' | 'hanning';

export interface EmphasizerParams {
  shiftX: number;
  shiftY: number;
  expU: number;
  expV: number;
  stretchFactor: number;
  mirrorAxis: 'horizontal' | 'vertical' | 'both';
  evenOddType: 'even' | 'odd';
  rotateAngle: number;
  diffDirection: 'x' | 'y' | 'both';
  intDirection: 'x' | 'y' | 'both';
  windowType: WindowType;
  windowWidthRatio: number;
  windowHeightRatio: number;
  windowSigma: number;
  ftCount: number; // This applies to ANY selected action!
  applyInFrequency: boolean;
}

export type AppMode = 'mixer' | 'emphasizer';
