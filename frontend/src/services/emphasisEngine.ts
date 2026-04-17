/**
 * emphasisEngine.ts — Optimized client-side emphasis transforms.
 *
 * Uses Float64Array throughout to avoid GC pressure from object allocations.
 * Convolution uses separable passes for O(n*k) instead of O(n*k²).
 * FFT operations use the flat-array API to avoid Complex[] ↔ Float64Array conversion.
 */

import type { Complex } from './fftEngine';
import { fft2dFromFlat, ifft2dFromFlat, fftShiftFlat, complexAbs } from './fftEngine';

export interface ComplexImage {
  real: Float64Array;
  imag: Float64Array;
  width: number;
  height: number;
}

export function createComplexFromGrayscale(pixels: number[], w: number, h: number): ComplexImage {
  const real = new Float64Array(w * h);
  const imag = new Float64Array(w * h);
  for (let i = 0; i < pixels.length; i++) real[i] = pixels[i];
  return { real, imag, width: w, height: h };
}

export function complexImageToPixels(img: ComplexImage, component: 'magnitude' | 'phase' | 'real' | 'imaginary'): number[] {
  const n = img.width * img.height;
  const raw = new Float64Array(n);

  switch (component) {
    case 'magnitude':
      for (let i = 0; i < n; i++) raw[i] = Math.sqrt(img.real[i] ** 2 + img.imag[i] ** 2);
      break;
    case 'phase': {
      const result = new Array(n);
      for (let i = 0; i < n; i++) {
        const v = Math.atan2(img.imag[i], img.real[i]);
        result[i] = Math.round(((v + Math.PI) / (2 * Math.PI)) * 255);
      }
      return result;
    }
    case 'real':
      for (let i = 0; i < n; i++) raw[i] = img.real[i];
      break;
    case 'imaginary':
      for (let i = 0; i < n; i++) raw[i] = img.imag[i];
      break;
  }

  let min = Infinity, max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (raw[i] < min) min = raw[i];
    if (raw[i] > max) max = raw[i];
  }
  const range = max - min || 1;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = Math.round(((raw[i] - min) / range) * 255);
  }
  return result;
}

export function shiftImage(img: ComplexImage, dx: number, dy: number): ComplexImage {
  const { width: w, height: h } = img;
  const n = w * h;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);

  for (let r = 0; r < h; r++) {
    const sr = ((r - dy) % h + h) % h;
    for (let c = 0; c < w; c++) {
      const sc = ((c - dx) % w + w) % w;
      const dstIdx = r * w + c;
      const srcIdx = sr * w + sc;
      outReal[dstIdx] = img.real[srcIdx];
      outImag[dstIdx] = img.imag[srcIdx];
    }
  }
  return { real: outReal, imag: outImag, width: w, height: h };
}

export function multiplyByExp(img: ComplexImage, u0: number, v0: number): ComplexImage {
  const { width: w, height: h } = img;
  const n = w * h;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);

  const twoPiOverW = 2 * Math.PI * u0 / w;
  const twoPiOverH = 2 * Math.PI * v0 / h;

  for (let r = 0; r < h; r++) {
    const phaseY = twoPiOverH * r;
    for (let c = 0; c < w; c++) {
      const idx = r * w + c;
      const phase = twoPiOverW * c + phaseY;
      const expRe = Math.cos(phase);
      const expIm = Math.sin(phase);
      outReal[idx] = img.real[idx] * expRe - img.imag[idx] * expIm;
      outImag[idx] = img.real[idx] * expIm + img.imag[idx] * expRe;
    }
  }
  return { real: outReal, imag: outImag, width: w, height: h };
}

export function stretchImage(img: ComplexImage, factor: number): ComplexImage {
  const { width: w, height: h } = img;
  const nw = Math.max(1, Math.round(w * factor));
  const nh = Math.max(1, Math.round(h * factor));
  const n = nw * nh;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);
  const invFactor = 1 / factor;

  for (let r = 0; r < nh; r++) {
    const sr = Math.min(Math.floor(r * invFactor), h - 1);
    const srcRow = sr * w;
    for (let c = 0; c < nw; c++) {
      const sc = Math.min(Math.floor(c * invFactor), w - 1);
      const dstIdx = r * nw + c;
      const srcIdx = srcRow + sc;
      outReal[dstIdx] = img.real[srcIdx];
      outImag[dstIdx] = img.imag[srcIdx];
    }
  }
  return { real: outReal, imag: outImag, width: nw, height: nh };
}

export function mirrorImage(img: ComplexImage, axis: 'horizontal' | 'vertical' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const outReal = new Float64Array(img.real);
  const outImag = new Float64Array(img.imag);

  if (axis === 'horizontal' || axis === 'both') {
    const mid = Math.floor(w / 2);
    for (let r = 0; r < h; r++) {
      const rowOff = r * w;
      for (let c = mid; c < w; c++) {
        const srcC = w - 1 - c;
        outReal[rowOff + c] = outReal[rowOff + srcC];
        outImag[rowOff + c] = outImag[rowOff + srcC];
      }
    }
  }
  if (axis === 'vertical' || axis === 'both') {
    const mid = Math.floor(h / 2);
    for (let r = mid; r < h; r++) {
      const srcR = h - 1 - r;
      const dstOff = r * w;
      const srcOff = srcR * w;
      for (let c = 0; c < w; c++) {
        outReal[dstOff + c] = outReal[srcOff + c];
        outImag[dstOff + c] = outImag[srcOff + c];
      }
    }
  }
  return { real: outReal, imag: outImag, width: w, height: h };
}

export function makeEvenOdd(img: ComplexImage, type: 'even' | 'odd'): ComplexImage {
  const { width: w, height: h } = img;
  const nw = w * 2;
  const nh = h * 2;
  const n = nw * nh;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);
  const sign = type === 'even' ? 1 : -1;

  for (let r = 0; r < nh; r++) {
    for (let c = 0; c < nw; c++) {
      const cr = r < h ? r : (nh - r);
      const cc = c < w ? c : (nw - c);
      const isFlipped = (r >= h || c >= w);

      const sr = Math.min(Math.abs(cr), h - 1);
      const sc = Math.min(Math.abs(cc), w - 1);

      const mul = isFlipped ? sign : 1;
      const dstIdx = r * nw + c;
      const srcIdx = sr * w + sc;
      outReal[dstIdx] = img.real[srcIdx] * mul;
      outImag[dstIdx] = img.imag[srcIdx] * mul;
    }
  }
  return { real: outReal, imag: outImag, width: nw, height: nh };
}

export function rotateImage(img: ComplexImage, angleDeg: number): ComplexImage {
  const { width: w, height: h } = img;
  const rad = (angleDeg * Math.PI) / 180;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  const corners = [[-w/2, -h/2], [w/2, -h/2], [-w/2, h/2], [w/2, h/2]];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of corners) {
    const rx = x * cosA - y * sinA;
    const ry = x * sinA + y * cosA;
    minX = Math.min(minX, rx); maxX = Math.max(maxX, rx);
    minY = Math.min(minY, ry); maxY = Math.max(maxY, ry);
  }

  const nw = Math.ceil(maxX - minX);
  const nh = Math.ceil(maxY - minY);
  const n = nw * nh;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);

  const cx = w / 2, cy = h / 2;
  const ncx = nw / 2, ncy = nh / 2;

  for (let r = 0; r < nh; r++) {
    const dy = r - ncy;
    for (let c = 0; c < nw; c++) {
      const dx = c - ncx;
      const sx = dx * cosA + dy * sinA + cx;
      const sy = -dx * sinA + dy * cosA + cy;
      const si = Math.round(sy), sj = Math.round(sx);
      if (si >= 0 && si < h && sj >= 0 && sj < w) {
        const dstIdx = r * nw + c;
        const srcIdx = si * w + sj;
        outReal[dstIdx] = img.real[srcIdx];
        outImag[dstIdx] = img.imag[srcIdx];
      }
    }
  }
  return { real: outReal, imag: outImag, width: nw, height: nh };
}

export function differentiateImage(img: ComplexImage, direction: 'x' | 'y' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const n = w * h;
  const outReal = new Float64Array(n);
  const outImag = new Float64Array(n);

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = r * w + c;
      let dxRe = 0, dxIm = 0, dyRe = 0, dyIm = 0;

      if (direction === 'x' || direction === 'both') {
        const right = c < w - 1 ? r * w + c + 1 : idx;
        const left = c > 0 ? r * w + c - 1 : idx;
        dxRe = (img.real[right] - img.real[left]) * 0.5;
        dxIm = (img.imag[right] - img.imag[left]) * 0.5;
      }
      if (direction === 'y' || direction === 'both') {
        const down = r < h - 1 ? (r + 1) * w + c : idx;
        const up = r > 0 ? (r - 1) * w + c : idx;
        dyRe = (img.real[down] - img.real[up]) * 0.5;
        dyIm = (img.imag[down] - img.imag[up]) * 0.5;
      }

      if (direction === 'both') {
        outReal[idx] = Math.sqrt(dxRe * dxRe + dyRe * dyRe);
        outImag[idx] = Math.sqrt(dxIm * dxIm + dyIm * dyIm);
      } else if (direction === 'x') {
        outReal[idx] = dxRe; outImag[idx] = dxIm;
      } else {
        outReal[idx] = dyRe; outImag[idx] = dyIm;
      }
    }
  }
  return { real: outReal, imag: outImag, width: w, height: h };
}

export function integrateImage(img: ComplexImage, direction: 'x' | 'y' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const outReal = new Float64Array(img.real);
  const outImag = new Float64Array(img.imag);

  if (direction === 'x' || direction === 'both') {
    for (let r = 0; r < h; r++) {
      const rowOff = r * w;
      for (let c = 1; c < w; c++) {
        outReal[rowOff + c] += outReal[rowOff + c - 1];
        outImag[rowOff + c] += outImag[rowOff + c - 1];
      }
    }
  }
  if (direction === 'y' || direction === 'both') {
    for (let c = 0; c < w; c++) {
      for (let r = 1; r < h; r++) {
        outReal[r * w + c] += outReal[(r - 1) * w + c];
        outImag[r * w + c] += outImag[(r - 1) * w + c];
      }
    }
  }
  return { real: outReal, imag: outImag, width: w, height: h };
}

export type WindowType = 'rectangular' | 'gaussian' | 'hamming' | 'hanning';

export interface WindowParams {
  type: WindowType;
  kernelWidth: number;
  kernelHeight: number;
  strideX: number;
  strideY: number;
  sigma?: number;
  mode?: 'same' | 'valid';
}

function generateWindow1D(n: number, type: WindowType, sigma: number): Float64Array {
  const win = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    switch (type) {
      case 'rectangular':
        win[i] = 1;
        break;
      case 'gaussian': {
        const center = n / 2;
        const std = Math.max(0.1, sigma * n / 4);
        const dist = i - center;
        win[i] = Math.exp(-(dist * dist) / (2 * std * std));
        break;
      }
      case 'hamming':
        win[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * (i / Math.max(1, n - 1)));
        break;
      case 'hanning':
        win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * (i / Math.max(1, n - 1))));
        break;
    }
  }
  return win;
}

/**
 * Separable 2D convolution: applies 1D horizontal pass then 1D vertical pass.
 * O(n * (kw + kh)) instead of O(n * kw * kh).
 */
function convolve2dSeparable(
  input: Float64Array, inW: number, inH: number,
  winX: Float64Array, winY: Float64Array,
  kw: number, kh: number,
  strideX: number, strideY: number, mode: 'same' | 'valid',
): { data: Float64Array; outW: number; outH: number } {
  // Normalize the window (we do it inline since it's separable)
  let sumX = 0, sumY = 0;
  for (let i = 0; i < kw; i++) sumX += winX[i];
  for (let i = 0; i < kh; i++) sumY += winY[i];
  // Normalize so the overall sum = 1: each window gets sqrt-normalized
  // Actually, we normalize by sumX * sumY at the end
  const totalSum = sumX * sumY;
  const invSum = totalSum > 0 ? 1 / totalSum : 1;

  let padX = 0, padY = 0;
  if (mode === 'same') {
    padX = Math.floor(kw / 2);
    padY = Math.floor(kh / 2);
  }

  // Pass 1: Horizontal convolution with winX
  const midH = inH;
  const midW = mode === 'same' ? inW : Math.max(1, inW - kw + 1);
  const mid = new Float64Array(midH * midW);

  for (let r = 0; r < midH; r++) {
    for (let c = 0; c < midW; c++) {
      let sum = 0;
      const startC = c - padX;
      for (let k = 0; k < kw; k++) {
        const srcC = startC + k;
        if (srcC >= 0 && srcC < inW) {
          sum += input[r * inW + srcC] * winX[k];
        }
      }
      mid[r * midW + c] = sum;
    }
  }

  // Pass 2: Vertical convolution with winY
  const preStrideH = mode === 'same' ? midH : Math.max(1, midH - kh + 1);
  const preStrideW = midW;

  const preStride = new Float64Array(preStrideH * preStrideW);

  for (let c = 0; c < preStrideW; c++) {
    for (let r = 0; r < preStrideH; r++) {
      let sum = 0;
      const startR = r - padY;
      for (let k = 0; k < kh; k++) {
        const srcR = startR + k;
        if (srcR >= 0 && srcR < midH) {
          sum += mid[srcR * midW + c] * winY[k];
        }
      }
      preStride[r * preStrideW + c] = sum * invSum;
    }
  }

  // Apply stride (subsample)
  const sx = Math.max(1, strideX);
  const sy = Math.max(1, strideY);
  if (sx === 1 && sy === 1) {
    return { data: preStride, outW: preStrideW, outH: preStrideH };
  }

  const outH = Math.ceil(preStrideH / sy);
  const outW = Math.ceil(preStrideW / sx);
  const out = new Float64Array(outH * outW);
  for (let r = 0; r < outH; r++) {
    for (let c = 0; c < outW; c++) {
      out[r * outW + c] = preStride[(r * sy) * preStrideW + (c * sx)];
    }
  }
  return { data: out, outW, outH };
}

export function applyWindow(img: ComplexImage, params: WindowParams): ComplexImage {
  const { width: w, height: h } = img;
  const sigma = params.sigma ?? 1.0;
  const mode = params.mode ?? 'same';
  const sx = Math.max(1, params.strideX);
  const sy = Math.max(1, params.strideY);

  // Ensure odd kernel sizes
  const kw = Math.max(1, params.kernelWidth) | 1;
  const kh = Math.max(1, params.kernelHeight) | 1;

  // Build 1D windows
  const winX = generateWindow1D(kw, params.type, sigma);
  const winY = generateWindow1D(kh, params.type, sigma);

  // Use separable convolution for both real and imaginary parts
  const realResult = convolve2dSeparable(img.real, w, h, winX, winY, kw, kh, sx, sy, mode);
  const imagResult = convolve2dSeparable(img.imag, w, h, winX, winY, kw, kh, sx, sy, mode);

  return {
    real: realResult.data,
    imag: imagResult.data,
    width: realResult.outW,
    height: realResult.outH,
  };
}

export function applyMultipleFT(img: ComplexImage, count: number): ComplexImage {
  let curRe = img.real;
  let curIm = img.imag;
  let curW = img.width;
  let curH = img.height;

  for (let i = 0; i < count; i++) {
    const result = fft2dFromFlat(curRe, curIm, curW, curH);
    curRe = result.re;
    curIm = result.im;
    curW = result.cols;
    curH = result.rows;
  }

  return { real: curRe, imag: curIm, width: curW, height: curH };
}

export function computeFT(img: ComplexImage): ComplexImage {
  const result = fft2dFromFlat(img.real, img.imag, img.width, img.height);
  return { real: result.re, imag: result.im, width: result.cols, height: result.rows };
}

export function computeIFT(img: ComplexImage): ComplexImage {
  const result = ifft2dFromFlat(img.real, img.imag, img.width, img.height);
  return { real: result.re, imag: result.im, width: result.cols, height: result.rows };
}

export function complexToDisplayPixels(
  img: ComplexImage,
  component: 'magnitude' | 'phase' | 'real' | 'imaginary'
): number[] {
  return complexImageToPixels(img, component);
}

export function shiftComplexImage(img: ComplexImage): ComplexImage {
  const { width: w, height: h } = img;
  const result = fftShiftFlat(img.real, img.imag, h, w);
  return { real: result.re, imag: result.im, width: w, height: h };
}

export function loadFileAsComplexImage(file: File): Promise<{ img: ComplexImage; pixels: number[] }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const htmlImg = new Image();
      htmlImg.onload = () => {
        const w = htmlImg.naturalWidth;
        const h = htmlImg.naturalHeight;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(htmlImg, 0, 0);
        const data = ctx.getImageData(0, 0, w, h);
        const pixels: number[] = [];
        for (let i = 0; i < data.data.length; i += 4) {
          pixels.push(Math.round(0.299 * data.data[i] + 0.587 * data.data[i + 1] + 0.114 * data.data[i + 2]));
        }
        const img = createComplexFromGrayscale(pixels, w, h);
        resolve({ img, pixels });
      };
      htmlImg.onerror = reject;
      htmlImg.src = reader.result as string;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function complexImageToPngBase64(img: ComplexImage, component: 'magnitude' | 'phase' | 'real' | 'imaginary'): string {
  const pixels = complexToDisplayPixels(img, component);
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(img.width, img.height);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.max(0, Math.min(255, pixels[i]));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function complexImageToFTPngBase64(img: ComplexImage, component: 'magnitude' | 'phase' | 'real' | 'imaginary'): string {
  const shifted = shiftComplexImage(img);
  return complexImageToPngBase64(shifted, component);
}

export { complexAbs };
