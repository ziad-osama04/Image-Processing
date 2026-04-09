import { useState, useCallback, useRef, useEffect } from 'react';
import type { EmphasizerAction, EmphasizerParams, FTComponent } from '../types/image';
import { useSession } from '../hooks/useSession';
import * as imageApi from '../services/imageApi';
import * as emphasizerApi from '../services/emphasizerApi';
import { ProgressBar } from './ProgressBar';
import { OperationSelector } from './OperationSelector';
import { DomainIndicator } from './DomainIndicator';
import { DynamicParameterPanel } from './DynamicParameterPanel';
import {
  type ComplexImage,
  loadFileAsComplexImage,
  shiftImage,
  multiplyByExp,
  stretchImage,
  mirrorImage,
  makeEvenOdd,
  rotateImage,
  differentiateImage,
  integrateImage,
  applyWindow,
  applyMultipleFT,
  computeFT,
  computeIFT,
  complexImageToPngBase64,
  complexImageToFTPngBase64,
} from '../services/emphasisEngine';
import { useMouseDrag } from '../hooks/useMouseDrag';

function EmphasisViewport({ label, imageSrc, component, onComponentChange, allowUpload, onLoad }: {
  label: string;
  imageSrc: string | null;
  component: FTComponent | 'magnitude';
  onComponentChange: (c: FTComponent | 'magnitude') => void;
  allowUpload?: boolean;
  onLoad?: () => void;
}) {
  const { brightness, contrast, isDragging, handlers } = useMouseDrag();
  return (
    <div className="viewport-component">
      <div className="viewport-header">
        <span className="viewport-label">{label}</span>
        <select
          value={component}
          onChange={e => onComponentChange(e.target.value as FTComponent)}
          className="viewport-dropdown"
          disabled={!imageSrc}
        >
          <option value="magnitude">Magnitude</option>
          <option value="phase">Phase</option>
          <option value="real">Real</option>
          <option value="imaginary">Imaginary</option>
        </select>
      </div>
      <div
        className={`viewport-canvas ${isDragging ? 'dragging' : ''} ${allowUpload ? 'clickable' : ''}`}
        onDoubleClick={allowUpload ? onLoad : undefined}
        onMouseDown={handlers.onMouseDown}
        onMouseMove={handlers.onMouseMove}
        onMouseUp={handlers.onMouseUp}
        onMouseLeave={handlers.onMouseLeave}
        style={{ filter: `brightness(${1 + brightness}) contrast(${contrast})` }}
      >
        {imageSrc ? (
          <img src={imageSrc} alt={label} className="viewport-image" draggable={false} />
        ) : (
          <div className="viewport-empty">
            <svg className="viewport-empty-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="viewport-empty-text">{allowUpload ? 'Double-click to load' : 'Result'}</span>
          </div>
        )}
      </div>
      <div className="viewport-footer">
        <span className="viewport-filename">{imageSrc ? label : 'No image'}</span>
        <span>B:{brightness.toFixed(2)} C:{contrast.toFixed(2)}</span>
      </div>
    </div>
  );
}

const DEFAULT_PARAMS: EmphasizerParams = {
  shiftX: 10, shiftY: 10,
  expU: 5, expV: 5,
  stretchFactor: 1.5,
  mirrorAxis: 'horizontal',
  evenOddType: 'even',
  rotateAngle: 45,
  diffDirection: 'both',
  intDirection: 'x',
  windowType: 'gaussian',
  windowKernelWidth: 5,
  windowKernelHeight: 5,
  windowStrideX: 1,
  windowStrideY: 1,
  windowSigma: 1.0,
  windowMode: 'same',
  ftCount: 0,
  applyInFrequency: false,
};

export function EmphasizerWorkspace() {
  const { sessionId } = useSession();
  const [useBackend, setUseBackend] = useState<boolean>(true);
  const [progress, setProgress] = useState<number>(0);
  const [showProgress, setShowProgress] = useState<boolean>(false);
  const backendCancelRef = useRef<(() => void) | null>(null);

  const [action, setAction] = useState<EmphasizerAction>('shift');
  const [params, setParams] = useState<EmphasizerParams>(DEFAULT_PARAMS);
  const [originalImage, setOriginalImage] = useState<ComplexImage | null>(null);
  const [originalPixels, setOriginalPixels] = useState<number[] | null>(null);
  const [processing, setProcessing] = useState(false);
  const [latestRequestId, setLatestRequestId] = useState<number | null>(null);

  // Display images (data URLs)
  const [origSpatial, setOrigSpatial] = useState<string | null>(null);
  const [modSpatial, setModSpatial] = useState<string | null>(null);
  const [origFT, setOrigFT] = useState<string | null>(null);
  const [modFT, setModFT] = useState<string | null>(null);

  // Independent viewport component selection
  const [compOrigSpatial, setCompOrigSpatial] = useState<FTComponent | 'magnitude'>('magnitude');
  const [compModSpatial, setCompModSpatial] = useState<FTComponent | 'magnitude'>('magnitude');
  const [compOrigFT, setCompOrigFT] = useState<FTComponent | 'magnitude'>('magnitude');
  const [compModFT, setCompModFT] = useState<FTComponent | 'magnitude'>('magnitude');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const updateParam = useCallback(<K extends keyof EmphasizerParams>(key: K, value: EmphasizerParams[K]) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleLoadImage = useCallback(() => fileInputRef.current?.click(), []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      if (sessionId) await imageApi.uploadImage(sessionId, 0, file);
      const { img, pixels } = await loadFileAsComplexImage(file);
      setOriginalImage(img);
      setOriginalPixels(pixels);
    } catch (err) {
      console.error('Failed to load image:', err);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [sessionId]);

  const handleApplyBackend = useCallback(async () => {
    if (!sessionId || !originalImage) return;

    if (backendCancelRef.current) {
      backendCancelRef.current();
      backendCancelRef.current = null;
    }

    try {
      setShowProgress(true);
      setProgress(0);

      const request: emphasizerApi.TransformRequest = {
        operation: action,
        params: params as any,
        domain: params.applyInFrequency ? 'frequency' : 'spatial',
        slot: 0,
      };

      const { request_id } = await emphasizerApi.applyTransform(sessionId, request);

      const cancelProgress = emphasizerApi.connectProgress(
        sessionId,
        (event) => { if (event.request_id === request_id) setProgress(event.progress); },
        async () => {
          try {
            setLatestRequestId(request_id);
            // Immediately fetch the exact currently requested components instead of default previews
            const resSpatial = await emphasizerApi.getTransformComponent(sessionId, request_id, 'spatial', compModSpatial);
            const resFt = await emphasizerApi.getTransformComponent(sessionId, request_id, 'frequency', compModFT);
            setModSpatial(`data:image/png;base64,${resSpatial.image}`);
            setModFT(`data:image/png;base64,${resFt.image}`);
          } catch (err) {
            console.error('Failed to get backend components:', err);
          } finally {
            setTimeout(() => setShowProgress(false), 500);
          }
        },
        (err) => {
          console.error('Progress stream error:', err);
          setShowProgress(false);
        }
      );
      backendCancelRef.current = cancelProgress;
    } catch (err) {
      console.error('Backend execution failed:', err);
      setShowProgress(false);
    }
  }, [sessionId, originalImage, action, params, compModSpatial, compModFT]);

  // Fetch updated Backend Spatial component if user changes dropdown
  useEffect(() => {
    if (!useBackend || !latestRequestId || !sessionId) return;
    emphasizerApi.getTransformComponent(sessionId, latestRequestId, 'spatial', compModSpatial)
      .then(res => setModSpatial(`data:image/png;base64,${res.image}`))
      .catch(console.error);
  }, [compModSpatial, useBackend, latestRequestId, sessionId]);

  // Fetch updated Backend FT component if user changes dropdown
  useEffect(() => {
    if (!useBackend || !latestRequestId || !sessionId) return;
    emphasizerApi.getTransformComponent(sessionId, latestRequestId, 'frequency', compModFT)
      .then(res => setModFT(`data:image/png;base64,${res.image}`))
      .catch(console.error);
  }, [compModFT, useBackend, latestRequestId, sessionId]);

  // Primary rendering effect (Local math & Original views)
  useEffect(() => {
    if (!originalImage || !originalPixels) return;
    setProcessing(true);

    const timeoutId = setTimeout(() => {
      try {
        setOrigSpatial(complexImageToPngBase64(originalImage, compOrigSpatial));
        const origFFT = computeFT(originalImage);
        setOrigFT(complexImageToFTPngBase64(origFFT, compOrigFT));

        if (useBackend) {
          setProcessing(false);
          return; // Let handleApplyBackend orchestrate backend fetching
        }

        let modified: ComplexImage;
        let targetImage = params.applyInFrequency ? origFFT : originalImage;

        switch (action) {
          case 'shift': modified = shiftImage(targetImage, params.shiftX, params.shiftY); break;
          case 'complex-exponential': modified = multiplyByExp(targetImage, params.expU, params.expV); break;
          case 'stretch': modified = stretchImage(targetImage, params.stretchFactor); break;
          case 'mirror': modified = mirrorImage(targetImage, params.mirrorAxis); break;
          case 'even-odd': modified = makeEvenOdd(targetImage, params.evenOddType); break;
          case 'rotate': modified = rotateImage(targetImage, params.rotateAngle); break;
          case 'differentiate': modified = differentiateImage(targetImage, params.diffDirection); break;
          case 'integrate': modified = integrateImage(targetImage, params.intDirection); break;
          case 'window':
            modified = applyWindow(targetImage, {
              type: params.windowType,
              kernelWidth: params.windowKernelWidth,
              kernelHeight: params.windowKernelHeight,
              strideX: params.windowStrideX,
              strideY: params.windowStrideY,
              sigma: params.windowSigma,
              mode: params.windowMode,
            });
            break;
          default: modified = targetImage;
        }

        if (params.ftCount > 0) {
          modified = applyMultipleFT(modified, params.ftCount);
        }

        if (params.applyInFrequency) {
          setModFT(complexImageToFTPngBase64(modified, compModFT));
          const ifftResult = computeIFT(modified);
          setModSpatial(complexImageToPngBase64(ifftResult, compModSpatial));
        } else {
          setModSpatial(complexImageToPngBase64(modified, compModSpatial));
          const modFFT = computeFT(modified);
          setModFT(complexImageToFTPngBase64(modFFT, compModFT));
        }
      } catch (err) {
        console.error('Emphasis computation error:', err);
      } finally {
        setProcessing(false);
      }
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [originalImage, originalPixels, action, params, compOrigSpatial, compModSpatial, compOrigFT, compModFT, useBackend]);

  return (
    <div className="workspace">
      <header className="workspace-header">
        <div>
          <h1 className="workspace-title">FT Properties Emphasizer</h1>
          <p className="workspace-subtitle">Explore Fourier Transform properties and duality</p>
        </div>
      </header>

      <main className="workspace-main emphasizer-layout">
        <aside className="action-panel">
          <OperationSelector action={action} onChange={setAction} />
          <DomainIndicator applyInFrequency={params.applyInFrequency} onChange={v => updateParam('applyInFrequency', v)} />

          <div className="action-panel-divider" />
          <DynamicParameterPanel action={action} params={params} onParamChange={updateParam} />

          {/* Multiple FT is applied ON TOP of the other actions */}
          <div className="action-panel-divider" />
          <div className="action-panel-section">
            <label className="action-label">Additional Operations</label>
            <div className="param-slider">
              <label className="param-slider-label">Repeated FT Count</label>
              <div className="param-slider-row">
                <input type="range" min={0} max={10} step={1} value={params.ftCount} onChange={e => updateParam('ftCount', parseInt(e.target.value))} className="mixer-slider" />
                <input type="number" min={0} max={10} step={1} value={params.ftCount} onChange={e => updateParam('ftCount', parseInt(e.target.value) || 0)} className="param-number-input" />
              </div>
            </div>
          </div>

          <div className="emphasizer-backend-controls" style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label className="backend-toggle-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <input type="checkbox" checked={useBackend} onChange={e => setUseBackend(e.target.checked)} className="backend-toggle-checkbox" />
              <span style={{ fontSize: '11px', fontWeight: 'bold' }}>Use Backend (Threaded)</span>
            </label>
            {useBackend && (
              <>
                <button
                  className="mixer-mix-btn"
                  onClick={handleApplyBackend}
                  disabled={!originalImage || showProgress}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  Apply to Backend
                </button>
                <ProgressBar progress={progress} visible={showProgress} />
              </>
            )}
          </div>
        </aside>

        <div className="emphasizer-viewports">
          {processing && (
            <div className="emphasizer-processing">
              <div className="spinner" />
              <span>Processing...</span>
            </div>
          )}
          <div className="emphasizer-grid">
            <EmphasisViewport
              label={params.applyInFrequency ? "Original Spatial" : "Original Spatial"}
              imageSrc={origSpatial}
              component={compOrigSpatial}
              onComponentChange={setCompOrigSpatial}
              allowUpload={true}
              onLoad={handleLoadImage}
            />
            <EmphasisViewport
              label={params.applyInFrequency ? "Result Spatial (IFFT)" : "Modified Spatial"}
              imageSrc={modSpatial}
              component={compModSpatial}
              onComponentChange={setCompModSpatial}
            />
            <EmphasisViewport
              label={params.applyInFrequency ? "Original FT" : "Original FT"}
              imageSrc={origFT}
              component={compOrigFT}
              onComponentChange={setCompOrigFT}
            />
            <EmphasisViewport
              label={params.applyInFrequency ? "Modified FT" : "Result FT"}
              imageSrc={modFT}
              component={compModFT}
              onComponentChange={setCompModFT}
            />
          </div>
        </div>
      </main>
      <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.bmp" onChange={handleFileChange} className="hidden" />
    </div>
  );
}
