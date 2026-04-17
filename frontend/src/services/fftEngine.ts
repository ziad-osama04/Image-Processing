/**
 * fftEngine.ts — Client-side 2D FFT engine for image processing.
 *
 * OPTIMIZED: Uses flat Float64Array pairs (re[], im[]) instead of
 * Complex object arrays to avoid GC pressure. In-place transforms
 * eliminate row/column slice copies.
 *
 * Provides:
 *  - Forward & Inverse 2D FFT (Cooley-Tukey radix-2)
 *  - Component extraction (magnitude, phase, real, imaginary)
 *  - FT mixing (weighted combination of multiple FFTs)
 *  - Region masking (inner/outer frequency selection)
 *  - Brightness/contrast adjustment
 *  - Pixel ↔ base64 PNG conversion
 */

// ── Complex helpers ─────────────────────────────────────────────────

export interface Complex {
  re: number;
  im: number;
}

export function complexAbs(z: Complex): number {
  return Math.sqrt(z.re * z.re + z.im * z.im);
}

export function complexAngle(z: Complex): number {
  return Math.atan2(z.im, z.re);
}

function complexFromPolar(mag: number, phase: number): Complex {
  return { re: mag * Math.cos(phase), im: mag * Math.sin(phase) };
}

// ── Power-of-2 padding ─────────────────────────────────────────────

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ── 1D FFT (iterative Cooley-Tukey) on flat arrays ──────────────────
// Operates on re[offset..offset+n-1] and im[offset..offset+n-1] with stride.
// For row transforms: stride=1, for column transforms: stride=cols.

function fft1dFlat(
  re: Float64Array, im: Float64Array,
  offset: number, n: number, stride: number,
  inverse: boolean
): void {
  if (n <= 1) return;

  // We need a temp buffer for bit-reversal + butterfly when stride != 1.
  // For efficiency, extract to a contiguous buffer, transform, then write back.
  const tmpRe = new Float64Array(n);
  const tmpIm = new Float64Array(n);

  // Copy strided data into contiguous temp
  for (let i = 0; i < n; i++) {
    const idx = offset + i * stride;
    tmpRe[i] = re[idx];
    tmpIm[i] = im[idx];
  }

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let t = tmpRe[i]; tmpRe[i] = tmpRe[j]; tmpRe[j] = t;
      t = tmpIm[i]; tmpIm[i] = tmpIm[j]; tmpIm[j] = t;
    }
  }

  // Butterfly stages
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curWRe = 1, curWIm = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;

        // complex multiply: w * odd
        const tRe = curWRe * tmpRe[oddIdx] - curWIm * tmpIm[oddIdx];
        const tIm = curWRe * tmpIm[oddIdx] + curWIm * tmpRe[oddIdx];

        tmpRe[oddIdx] = tmpRe[evenIdx] - tRe;
        tmpIm[oddIdx] = tmpIm[evenIdx] - tIm;
        tmpRe[evenIdx] = tmpRe[evenIdx] + tRe;
        tmpIm[evenIdx] = tmpIm[evenIdx] + tIm;

        // advance twiddle
        const newWRe = curWRe * wRe - curWIm * wIm;
        curWIm = curWRe * wIm + curWIm * wRe;
        curWRe = newWRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      tmpRe[i] /= n;
      tmpIm[i] /= n;
    }
  }

  // Write back
  for (let i = 0; i < n; i++) {
    const idx = offset + i * stride;
    re[idx] = tmpRe[i];
    im[idx] = tmpIm[i];
  }
}

// ── 2D FFT / IFFT on flat arrays ────────────────────────────────────

export interface FFTResult {
  data: Complex[];
  rows: number;
  cols: number;
}

/** Flat-array 2D FFT result for internal use */
export interface FlatFFTResult {
  re: Float64Array;
  im: Float64Array;
  rows: number;
  cols: number;
}

function fft2dFlat(
  inRe: Float64Array, inIm: Float64Array,
  width: number, height: number,
  inverse: boolean
): FlatFFTResult {
  const cols = nextPow2(width);
  const rows = nextPow2(height);
  const n = rows * cols;
  const re = new Float64Array(n);
  const im = new Float64Array(n);

  // Copy input (with zero-padding)
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      re[r * cols + c] = inRe[r * width + c];
      im[r * cols + c] = inIm[r * width + c];
    }
  }

  // Transform rows
  for (let r = 0; r < rows; r++) {
    fft1dFlat(re, im, r * cols, cols, 1, inverse);
  }
  // Transform columns
  for (let c = 0; c < cols; c++) {
    fft1dFlat(re, im, c, rows, cols, inverse);
  }

  return { re, im, rows, cols };
}

/** Forward 2D FFT. Input: real grayscale pixels, zero-padded to pow2. */
export function fft2d(pixels: number[], width: number, height: number): FFTResult {
  const inRe = new Float64Array(width * height);
  const inIm = new Float64Array(width * height);
  for (let i = 0; i < pixels.length; i++) inRe[i] = pixels[i];

  const flat = fft2dFlat(inRe, inIm, width, height, false);
  return flatToFFTResult(flat);
}

/** Forward 2D FFT on complex input. */
export function fft2dComplex(input: Complex[], width: number, height: number): FFTResult {
  const inRe = new Float64Array(width * height);
  const inIm = new Float64Array(width * height);
  for (let i = 0; i < input.length; i++) {
    inRe[i] = input[i].re;
    inIm[i] = input[i].im;
  }

  const flat = fft2dFlat(inRe, inIm, width, height, false);
  return flatToFFTResult(flat);
}

/** Forward 2D FFT on flat Float64Arrays (avoids Complex[] conversion). */
export function fft2dFromFlat(
  inRe: Float64Array, inIm: Float64Array,
  width: number, height: number
): FlatFFTResult {
  return fft2dFlat(inRe, inIm, width, height, false);
}

/** Inverse 2D FFT on flat Float64Arrays. */
export function ifft2dFromFlat(
  inRe: Float64Array, inIm: Float64Array,
  width: number, height: number
): FlatFFTResult {
  return fft2dFlat(inRe, inIm, width, height, true);
}

function flatToFFTResult(flat: FlatFFTResult): FFTResult {
  const n = flat.rows * flat.cols;
  const data: Complex[] = new Array(n);
  for (let i = 0; i < n; i++) {
    data[i] = { re: flat.re[i], im: flat.im[i] };
  }
  return { data, rows: flat.rows, cols: flat.cols };
}

/** Inverse 2D FFT → real pixel array (takes magnitude of result). */
export function ifft2d(fftResult: FFTResult): { pixels: number[]; width: number; height: number } {
  const { rows, cols } = fftResult;
  const inRe = new Float64Array(rows * cols);
  const inIm = new Float64Array(rows * cols);
  for (let i = 0; i < fftResult.data.length; i++) {
    inRe[i] = fftResult.data[i].re;
    inIm[i] = fftResult.data[i].im;
  }

  const flat = fft2dFlat(inRe, inIm, cols, rows, true);
  const pixels = new Array(flat.rows * flat.cols);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = Math.max(0, Math.min(255, Math.round(
      Math.sqrt(flat.re[i] * flat.re[i] + flat.im[i] * flat.im[i])
    )));
  }
  return { pixels, width: flat.cols, height: flat.rows };
}

/** Inverse 2D FFT → complex output (preserves phase). */
export function ifft2dComplex(fftResult: FFTResult): { data: Complex[]; width: number; height: number } {
  const { rows, cols } = fftResult;
  const inRe = new Float64Array(rows * cols);
  const inIm = new Float64Array(rows * cols);
  for (let i = 0; i < fftResult.data.length; i++) {
    inRe[i] = fftResult.data[i].re;
    inIm[i] = fftResult.data[i].im;
  }

  const flat = fft2dFlat(inRe, inIm, cols, rows, true);
  const data: Complex[] = new Array(flat.rows * flat.cols);
  for (let i = 0; i < data.length; i++) {
    data[i] = { re: flat.re[i], im: flat.im[i] };
  }
  return { data, width: flat.cols, height: flat.rows };
}

// ── FFT Shift ───────────────────────────────────────────────────────

export function fftShift(data: Complex[], rows: number, cols: number): Complex[] {
  const shifted = new Array<Complex>(rows * cols);
  const hr = rows >> 1;
  const hc = cols >> 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const sr = (r + hr) % rows;
      const sc = (c + hc) % cols;
      shifted[sr * cols + sc] = data[r * cols + c];
    }
  }
  return shifted;
}

export function fftShiftFlat(
  re: Float64Array, im: Float64Array,
  rows: number, cols: number
): { re: Float64Array; im: Float64Array } {
  const n = rows * cols;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const hr = rows >> 1;
  const hc = cols >> 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const sr = (r + hr) % rows;
      const sc = (c + hc) % cols;
      const srcIdx = r * cols + c;
      const dstIdx = sr * cols + sc;
      outRe[dstIdx] = re[srcIdx];
      outIm[dstIdx] = im[srcIdx];
    }
  }
  return { re: outRe, im: outIm };
}

export function ifftShift(data: Complex[], rows: number, cols: number): Complex[] {
  const shifted = new Array<Complex>(rows * cols);
  const hr = Math.ceil(rows / 2);
  const hc = Math.ceil(cols / 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const sr = (r + hr) % rows;
      const sc = (c + hc) % cols;
      shifted[sr * cols + sc] = data[r * cols + c];
    }
  }
  return shifted;
}

// ── Safe max/min (no stack overflow) ────────────────────────────────

function safeMax(arr: number[], initial = -Infinity): number {
  let max = initial;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i];
  }
  return max;
}

// ── Component Extraction ────────────────────────────────────────────

export function getMagnitude(fft: FFTResult): { pixels: number[]; width: number; height: number } {
  const shifted = fftShift(fft.data, fft.rows, fft.cols);
  const logScaled = new Array(shifted.length);
  for (let i = 0; i < shifted.length; i++) {
    logScaled[i] = Math.log1p(complexAbs(shifted[i]));
  }
  const max = safeMax(logScaled, 1e-10);
  const pixels = logScaled.map((v) => Math.round((v / max) * 255));
  return { pixels, width: fft.cols, height: fft.rows };
}

export function getPhase(fft: FFTResult): { pixels: number[]; width: number; height: number } {
  const shifted = fftShift(fft.data, fft.rows, fft.cols);
  const pixels = shifted.map((z) => Math.round(((complexAngle(z) + Math.PI) / (2 * Math.PI)) * 255));
  return { pixels, width: fft.cols, height: fft.rows };
}

export function getReal(fft: FFTResult): { pixels: number[]; width: number; height: number } {
  const shifted = fftShift(fft.data, fft.rows, fft.cols);
  const raw = shifted.map((z) => z.re);
  let absMax = 1e-10;
  for (let i = 0; i < raw.length; i++) {
    const a = Math.abs(raw[i]);
    if (a > absMax) absMax = a;
  }
  const pixels = raw.map((v) => Math.round(((v / absMax + 1) / 2) * 255));
  return { pixels, width: fft.cols, height: fft.rows };
}

export function getImaginary(fft: FFTResult): { pixels: number[]; width: number; height: number } {
  const shifted = fftShift(fft.data, fft.rows, fft.cols);
  const raw = shifted.map((z) => z.im);
  let absMax = 1e-10;
  for (let i = 0; i < raw.length; i++) {
    const a = Math.abs(raw[i]);
    if (a > absMax) absMax = a;
  }
  const pixels = raw.map((v) => Math.round(((v / absMax + 1) / 2) * 255));
  return { pixels, width: fft.cols, height: fft.rows };
}

/** Extract component from complex data (not FFT-shifted, for general complex images) */
export function getComponentFromComplex(
  data: Complex[],
  width: number,
  height: number,
  component: 'magnitude' | 'phase' | 'real' | 'imaginary'
): number[] {
  let raw: number[];
  switch (component) {
    case 'magnitude':
      raw = data.map((z) => Math.log1p(complexAbs(z)));
      break;
    case 'phase':
      return data.map((z) => Math.round(((complexAngle(z) + Math.PI) / (2 * Math.PI)) * 255));
    case 'real':
      raw = data.map((z) => z.re);
      break;
    case 'imaginary':
      raw = data.map((z) => z.im);
      break;
  }
  let absMax = 1e-10;
  for (let i = 0; i < raw.length; i++) {
    const a = Math.abs(raw[i]);
    if (a > absMax) absMax = a;
  }
  const _w = width; void _w;
  const _h = height; void _h;
  if (component === 'magnitude') {
    const max = safeMax(raw, 1e-10);
    return raw.map((v) => Math.round((v / max) * 255));
  }
  return raw.map((v) => Math.round(((v / absMax + 1) / 2) * 255));
}

// ── Region Masking ──────────────────────────────────────────────────

/**
 * Create a region mask for FFT data.
 * regionSize: 0–100 (percentage of total area)
 * regionType: 'inner' = keep low freq (center), 'outer' = keep high freq (edges)
 */
export function createRegionMask(
  rows: number,
  cols: number,
  regionSize: number,
  regionType: 'inner' | 'outer'
): Float32Array {
  const mask = new Float32Array(rows * cols);
  const fraction = regionSize / 100;
  const halfH = (rows * fraction) / 2;
  const halfW = (cols * fraction) / 2;
  const cr = rows / 2;
  const cc = cols / 2;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const inRect = Math.abs(r - cr) <= halfH && Math.abs(c - cc) <= halfW;
      mask[r * cols + c] = regionType === 'inner' ? (inRect ? 1 : 0) : (inRect ? 0 : 1);
    }
  }
  return mask;
}

// ── FT Mixing ───────────────────────────────────────────────────────

export interface MixerWeights {
  componentA: number; // magnitude or real weight
  componentB: number; // phase or imaginary weight
}

export type MixMode = 'mag-phase' | 'real-imag';

/**
 * Mix multiple FFT results using weighted combination.
 * Returns mixed FFT (NOT shifted — ready for IFFT).
 */
export function mixFFTs(
  ffts: FFTResult[],
  weights: MixerWeights[],
  mode: MixMode,
  regionSize: number,
  regionType: 'inner' | 'outer'
): FFTResult {
  if (ffts.length === 0) {
    return { data: [], rows: 0, cols: 0 };
  }

  // All FFTs must have same dimensions
  const { rows, cols } = ffts[0];
  const n = rows * cols;

  // Shift all FFTs to center
  const shiftedFFTs = ffts.map((f) => fftShift(f.data, rows, cols));

  // Create region mask
  const mask = createRegionMask(rows, cols, regionSize, regionType);

  // Mix
  const mixed: Complex[] = new Array(n);

  if (mode === 'mag-phase') {
    for (let i = 0; i < n; i++) {
      let mixedMag = 0;
      let mixedPhase = 0;
      for (let k = 0; k < shiftedFFTs.length; k++) {
        const z = shiftedFFTs[k][i];
        const m = mask[i];
        mixedMag += weights[k].componentA * complexAbs(z) * m;
        mixedPhase += weights[k].componentB * complexAngle(z) * m;
      }
      mixed[i] = complexFromPolar(mixedMag, mixedPhase);
    }
  } else {
    // real-imag mode
    for (let i = 0; i < n; i++) {
      let mixedRe = 0;
      let mixedIm = 0;
      for (let k = 0; k < shiftedFFTs.length; k++) {
        const z = shiftedFFTs[k][i];
        const m = mask[i];
        mixedRe += weights[k].componentA * z.re * m;
        mixedIm += weights[k].componentB * z.im * m;
      }
      mixed[i] = { re: mixedRe, im: mixedIm };
    }
  }

  // Shift back
  const unshifted = ifftShift(mixed, rows, cols);
  return { data: unshifted, rows, cols };
}

/**
 * Async mixing with progress reporting and cancellation.
 */
export async function mixFFTsAsync(
  ffts: FFTResult[],
  weights: MixerWeights[],
  mode: MixMode,
  regionSize: number,
  regionType: 'inner' | 'outer',
  onProgress: (pct: number) => void,
  signal: AbortSignal,
  simulateSlow: boolean
): Promise<{ pixels: number[]; width: number; height: number }> {
  onProgress(0);

  if (ffts.length === 0) return { pixels: [], width: 0, height: 0 };

  // Step 1: Mix FFTs (30%)
  await yieldToUI();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');

  if (simulateSlow) await delay(2000);
  onProgress(10);

  const mixedFFT = mixFFTs(ffts, weights, mode, regionSize, regionType);
  onProgress(30);

  await yieldToUI();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');

  if (simulateSlow) await delay(3000);
  onProgress(50);

  // Step 2: IFFT (80%)
  const result = ifft2d(mixedFFT);
  onProgress(80);

  await yieldToUI();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');

  if (simulateSlow) await delay(2000);
  onProgress(100);

  return result;
}

function yieldToUI(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Canvas Utilities ────────────────────────────────────────────────

export function pixelsToPngBase64(pixels: number[], width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.max(0, Math.min(255, pixels[i]));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png').replace('data:image/png;base64,', '');
}

export function applyBrightnessContrast(
  pixels: number[],
  width: number,
  height: number,
  brightness: number,
  contrast: number
): string {
  const adjusted = new Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    let p = (pixels[i] - 128) * contrast + 128 + brightness * 255;
    adjusted[i] = Math.max(0, Math.min(255, Math.round(p)));
  }
  return pixelsToPngBase64(adjusted, width, height);
}
