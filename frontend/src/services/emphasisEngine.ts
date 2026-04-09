import type { Complex } from './fftEngine';
import { fft2dComplex, ifft2dComplex, complexAbs } from './fftEngine';

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
  const raw = new Array(n);

  switch (component) {
    case 'magnitude':
      for (let i = 0; i < n; i++) raw[i] = Math.sqrt(img.real[i] ** 2 + img.imag[i] ** 2);
      break;
    case 'phase':
      for (let i = 0; i < n; i++) raw[i] = Math.atan2(img.imag[i], img.real[i]);
      return raw.map(v => Math.round(((v + Math.PI) / (2 * Math.PI)) * 255));
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
  return raw.map(v => Math.round(((v - min) / range) * 255));
}

function cloneComplex(img: ComplexImage): ComplexImage {
  return {
    real: new Float64Array(img.real),
    imag: new Float64Array(img.imag),
    width: img.width,
    height: img.height,
  };
}

export function shiftImage(img: ComplexImage, dx: number, dy: number): ComplexImage {
  const { width: w, height: h } = img;
  const out = cloneComplex(img);
  const tmpReal = new Float64Array(w * h);
  const tmpImag = new Float64Array(w * h);

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const sr = ((r - dy) % h + h) % h;
      const sc = ((c - dx) % w + w) % w;
      tmpReal[r * w + c] = img.real[sr * w + sc];
      tmpImag[r * w + c] = img.imag[sr * w + sc];
    }
  }
  out.real = tmpReal;
  out.imag = tmpImag;
  return out;
}

export function multiplyByExp(img: ComplexImage, u0: number, v0: number): ComplexImage {
  const { width: w, height: h } = img;
  const out = cloneComplex(img);

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = r * w + c;
      const phase = 2 * Math.PI * (u0 * c / w + v0 * r / h);
      const expRe = Math.cos(phase);
      const expIm = Math.sin(phase);
      out.real[idx] = img.real[idx] * expRe - img.imag[idx] * expIm;
      out.imag[idx] = img.real[idx] * expIm + img.imag[idx] * expRe;
    }
  }
  return out;
}

export function stretchImage(img: ComplexImage, factor: number): ComplexImage {
  const { width: w, height: h } = img;
  const nw = Math.max(1, Math.round(w * factor));
  const nh = Math.max(1, Math.round(h * factor));
  const out: ComplexImage = {
    real: new Float64Array(nw * nh),
    imag: new Float64Array(nw * nh),
    width: nw,
    height: nh,
  };

  for (let r = 0; r < nh; r++) {
    for (let c = 0; c < nw; c++) {
      const sr = Math.min(Math.floor(r / factor), h - 1);
      const sc = Math.min(Math.floor(c / factor), w - 1);
      out.real[r * nw + c] = img.real[sr * w + sc];
      out.imag[r * nw + c] = img.imag[sr * w + sc];
    }
  }
  return out;
}

export function mirrorImage(img: ComplexImage, axis: 'horizontal' | 'vertical' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const out = cloneComplex(img);
  
  if (axis === 'horizontal' || axis === 'both') {
    const mid = Math.floor(w / 2);
    for (let r = 0; r < h; r++) {
      for (let c = mid; c < w; c++) {
        const srcC = w - 1 - c;
        out.real[r * w + c] = out.real[r * w + srcC];
        out.imag[r * w + c] = out.imag[r * w + srcC];
      }
    }
  }
  if (axis === 'vertical' || axis === 'both') {
    const mid = Math.floor(h / 2);
    for (let r = mid; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const srcR = h - 1 - r;
        out.real[r * w + c] = out.real[srcR * w + c];
        out.imag[r * w + c] = out.imag[srcR * w + c];
      }
    }
  }
  return out;
}

export function makeEvenOdd(img: ComplexImage, type: 'even' | 'odd'): ComplexImage {
  const { width: w, height: h } = img;
  const nw = w * 2;
  const nh = h * 2;
  const out: ComplexImage = {
    real: new Float64Array(nw * nh),
    imag: new Float64Array(nw * nh),
    width: nw,
    height: nh,
  };

  const sign = type === 'even' ? 1 : -1;

  for (let r = 0; r < nh; r++) {
    for (let c = 0; c < nw; c++) {
      const cr = r < h ? r : (nh - r);
      const cc = c < w ? c : (nw - c);
      const isFlipped = (r >= h || c >= w);

      const sr = Math.min(Math.abs(cr), h - 1);
      const sc = Math.min(Math.abs(cc), w - 1);

      const mul = isFlipped ? sign : 1;
      out.real[r * nw + c] = img.real[sr * w + sc] * mul;
      out.imag[r * nw + c] = img.imag[sr * w + sc] * mul;
    }
  }
  return out;
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
  const out: ComplexImage = { real: new Float64Array(nw*nh), imag: new Float64Array(nw*nh), width: nw, height: nh };

  const cx = w / 2, cy = h / 2;
  const ncx = nw / 2, ncy = nh / 2;

  for (let r = 0; r < nh; r++) {
    for (let c = 0; c < nw; c++) {
      const dx = c - ncx, dy = r - ncy;
      const sx = dx * cosA + dy * sinA + cx;
      const sy = -dx * sinA + dy * cosA + cy;
      const si = Math.round(sy), sj = Math.round(sx);
      if (si >= 0 && si < h && sj >= 0 && sj < w) {
        out.real[r * nw + c] = img.real[si * w + sj];
        out.imag[r * nw + c] = img.imag[si * w + sj];
      }
    }
  }
  return out;
}

export function differentiateImage(img: ComplexImage, direction: 'x' | 'y' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const out: ComplexImage = { real: new Float64Array(w*h), imag: new Float64Array(w*h), width: w, height: h };

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = r * w + c;
      let dxRe = 0, dxIm = 0, dyRe = 0, dyIm = 0;

      if (direction === 'x' || direction === 'both') {
        const right = c < w - 1 ? r * w + c + 1 : idx;
        const left = c > 0 ? r * w + c - 1 : idx;
        dxRe = (img.real[right] - img.real[left]) / 2;
        dxIm = (img.imag[right] - img.imag[left]) / 2;
      }
      if (direction === 'y' || direction === 'both') {
        const down = r < h - 1 ? (r + 1) * w + c : idx;
        const up = r > 0 ? (r - 1) * w + c : idx;
        dyRe = (img.real[down] - img.real[up]) / 2;
        dyIm = (img.imag[down] - img.imag[up]) / 2;
      }

      if (direction === 'both') {
        out.real[idx] = Math.sqrt(dxRe * dxRe + dyRe * dyRe);
        out.imag[idx] = Math.sqrt(dxIm * dxIm + dyIm * dyIm);
      } else if (direction === 'x') {
        out.real[idx] = dxRe; out.imag[idx] = dxIm;
      } else {
        out.real[idx] = dyRe; out.imag[idx] = dyIm;
      }
    }
  }
  return out;
}

export function integrateImage(img: ComplexImage, direction: 'x' | 'y' | 'both'): ComplexImage {
  const { width: w, height: h } = img;
  const out = cloneComplex(img);

  if (direction === 'x' || direction === 'both') {
    for (let r = 0; r < h; r++) {
      for (let c = 1; c < w; c++) {
        out.real[r * w + c] += out.real[r * w + c - 1];
        out.imag[r * w + c] += out.imag[r * w + c - 1];
      }
    }
  }
  if (direction === 'y' || direction === 'both') {
    for (let c = 0; c < w; c++) {
      for (let r = 1; r < h; r++) {
        out.real[r * w + c] += out.real[(r - 1) * w + c];
        out.imag[r * w + c] += out.imag[(r - 1) * w + c];
      }
    }
  }
  return out;
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

function convolve2dChannel(
  input: Float64Array, inW: number, inH: number,
  kernel: Float64Array, kw: number, kh: number,
  strideX: number, strideY: number, mode: 'same' | 'valid',
): { data: Float64Array; outW: number; outH: number } {
  let padX = 0, padY = 0;
  if (mode === 'same') {
    padX = Math.floor(kw / 2);
    padY = Math.floor(kh / 2);
  }
  const paddedW = inW + 2 * padX;
  const paddedH = inH + 2 * padY;

  // Create zero-padded input
  const padded = new Float64Array(paddedW * paddedH);
  for (let r = 0; r < inH; r++) {
    for (let c = 0; c < inW; c++) {
      padded[(r + padY) * paddedW + (c + padX)] = input[r * inW + c];
    }
  }

  // Output dimensions
  const outH = Math.floor((paddedH - kh) / strideY) + 1;
  const outW = Math.floor((paddedW - kw) / strideX) + 1;
  const out = new Float64Array(outH * outW);

  // Sliding-window convolution
  for (let or_ = 0; or_ < outH; or_++) {
    const inR = or_ * strideY;
    for (let oc = 0; oc < outW; oc++) {
      const inC = oc * strideX;
      let sum = 0;
      for (let kr = 0; kr < kh; kr++) {
        for (let kc = 0; kc < kw; kc++) {
          sum += padded[(inR + kr) * paddedW + (inC + kc)] * kernel[kr * kw + kc];
        }
      }
      out[or_ * outW + oc] = sum;
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

  // Build 2D kernel (separable outer product)
  const winX = generateWindow1D(kw, params.type, sigma);
  const winY = generateWindow1D(kh, params.type, sigma);
  const kernel = new Float64Array(kh * kw);
  let kSum = 0;
  for (let r = 0; r < kh; r++) {
    for (let c = 0; c < kw; c++) {
      const v = winY[r] * winX[c];
      kernel[r * kw + c] = v;
      kSum += v;
    }
  }
  // Normalize kernel
  if (kSum > 0) {
    for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;
  }

  // Convolve real and imaginary parts separately
  const realResult = convolve2dChannel(img.real, w, h, kernel, kw, kh, sx, sy, mode);
  const imagResult = convolve2dChannel(img.imag, w, h, kernel, kw, kh, sx, sy, mode);

  return {
    real: realResult.data,
    imag: imagResult.data,
    width: realResult.outW,
    height: realResult.outH,
  };
}

export function applyMultipleFT(img: ComplexImage, count: number): ComplexImage {
  let current = img;
  for (let i = 0; i < count; i++) {
    const complexArr: Complex[] = new Array(current.width * current.height);
    for (let j = 0; j < complexArr.length; j++) {
      complexArr[j] = { re: current.real[j], im: current.imag[j] };
    }
    const fftResult = fft2dComplex(complexArr, current.width, current.height);

    current = {
      real: new Float64Array(fftResult.rows * fftResult.cols),
      imag: new Float64Array(fftResult.rows * fftResult.cols),
      width: fftResult.cols,
      height: fftResult.rows,
    };
    for (let j = 0; j < fftResult.data.length; j++) {
      current.real[j] = fftResult.data[j].re;
      current.imag[j] = fftResult.data[j].im;
    }
  }
  return current;
}

export function computeFT(img: ComplexImage): ComplexImage {
  const complexArr: Complex[] = new Array(img.width * img.height);
  for (let j = 0; j < complexArr.length; j++) {
    complexArr[j] = { re: img.real[j], im: img.imag[j] };
  }
  const fftResult = fft2dComplex(complexArr, img.width, img.height);
  const out: ComplexImage = {
    real: new Float64Array(fftResult.rows * fftResult.cols),
    imag: new Float64Array(fftResult.rows * fftResult.cols),
    width: fftResult.cols,
    height: fftResult.rows,
  };
  for (let j = 0; j < fftResult.data.length; j++) {
    out.real[j] = fftResult.data[j].re;
    out.imag[j] = fftResult.data[j].im;
  }
  return out;
}

export function computeIFT(img: ComplexImage): ComplexImage {
  const complexArr: Complex[] = new Array(img.width * img.height);
  for (let j = 0; j < complexArr.length; j++) {
    complexArr[j] = { re: img.real[j], im: img.imag[j] };
  }
  const result = ifft2dComplex({ data: complexArr, rows: img.height, cols: img.width });
  const out: ComplexImage = {
    real: new Float64Array(result.width * result.height),
    imag: new Float64Array(result.width * result.height),
    width: result.width,
    height: result.height,
  };
  for (let j = 0; j < result.data.length; j++) {
    out.real[j] = result.data[j].re;
    out.imag[j] = result.data[j].im;
  }
  return out;
}

export function complexToDisplayPixels(
  img: ComplexImage,
  component: 'magnitude' | 'phase' | 'real' | 'imaginary'
): number[] {
  return complexImageToPixels(img, component);
}

export function shiftComplexImage(img: ComplexImage): ComplexImage {
  const { width: w, height: h } = img;
  const hw = w >> 1, hh = h >> 1;
  const out: ComplexImage = {
    real: new Float64Array(w * h),
    imag: new Float64Array(w * h),
    width: w,
    height: h,
  };
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const sr = (r + hh) % h;
      const sc = (c + hw) % w;
      out.real[sr * w + sc] = img.real[r * w + c];
      out.imag[sr * w + sc] = img.imag[r * w + c];
    }
  }
  return out;
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
