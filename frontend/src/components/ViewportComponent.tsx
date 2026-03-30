/**
 * ViewportComponent: Reusable image/FT viewport.
 *
 * Single shared component for ALL viewports in the app (Constitution §III).
 * Supports: image display, FT component dropdown, mouse-drag B/C,
 * double-click browse, region overlay, and external image source.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { FTComponent } from '../types/image';
import { getImage, getFTComponent, uploadImage } from '../services/imageApi';
import { useMouseDrag } from '../hooks/useMouseDrag';

interface RegionOverlay {
  size: number; // 0–100 percentage
  type: 'inner' | 'outer';
}

interface ViewportComponentProps {
  sessionId: string;
  slot: number;
  label?: string;
  allowUpload?: boolean;
  onImageLoaded?: (slot: number, filename: string) => void;
  viewMode?: 'spatial' | FTComponent;
  hideDropdown?: boolean;
  regionOverlay?: RegionOverlay;
  /** If set, display this base64 image directly instead of fetching from API */
  externalSrc?: string | null;
}

export function ViewportComponent({
  sessionId,
  slot,
  label,
  allowUpload = true,
  onImageLoaded,
  viewMode,
  hideDropdown = false,
  regionOverlay,
  externalSrc,
}: ViewportComponentProps) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [filename, setFilename] = useState('');
  const [internalActiveComponent, setInternalActiveComponent] = useState<'spatial' | FTComponent>('spatial');
  const activeComponent = viewMode || internalActiveComponent;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { brightness, contrast, isDragging, handlers, reset } = useMouseDrag();
  const fetchIdRef = useRef(0);

  // When externalSrc is provided, use it directly
  useEffect(() => {
    if (externalSrc !== undefined) {
      setImageSrc(externalSrc);
    }
  }, [externalSrc]);

  // Fetch the display image from API (only when not using external source)
  const fetchDisplay = useCallback(
    async (component: 'spatial' | FTComponent, b: number, c: number) => {
      if (!sessionId || externalSrc !== undefined) return;
      const id = ++fetchIdRef.current;
      setLoading(true);
      setError(null);
      try {
        if (component === 'spatial') {
          const res = await getImage(sessionId, slot, b, c);
          if (fetchIdRef.current !== id) return;
          setImageSrc(`data:image/png;base64,${res.preview}`);
          setFilename(res.filename);
        } else {
          const res = await getFTComponent(sessionId, slot, component, b, c);
          if (fetchIdRef.current !== id) return;
          setImageSrc(`data:image/png;base64,${res.image}`);
        }
      } catch (err: unknown) {
        if (fetchIdRef.current !== id) return;
        if (err instanceof Error && !err.message.includes('404') && !err.message.includes('is empty')) {
          setError(err.message);
        }
      } finally {
        if (fetchIdRef.current === id) setLoading(false);
      }
    },
    [sessionId, slot, externalSrc]
  );

  // Re-fetch when component/session/B&C changes (debounced during drag)
  useEffect(() => {
    if (externalSrc !== undefined) return;
    if (isDragging) return; // don't fetch during active drag
    fetchDisplay(activeComponent, brightness, contrast);
  }, [sessionId, activeComponent, fetchDisplay, brightness, contrast, isDragging, externalSrc]);

  const handleComponentChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as 'spatial' | FTComponent;
      setInternalActiveComponent(value);
    },
    []
  );

  const handleDoubleClick = useCallback(() => {
    if (!allowUpload) return;
    fileInputRef.current?.click();
  }, [allowUpload]);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !sessionId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await uploadImage(sessionId, slot, file);
        setImageSrc(`data:image/png;base64,${res.preview}`);
        setFilename(res.filename);
        reset();
        onImageLoaded?.(slot, res.filename);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setLoading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    },
    [sessionId, slot, onImageLoaded, reset]
  );

  return (
    <div className="viewport-component">
      {/* Header */}
      <div className="viewport-header">
        <span className="viewport-label">
          {label || `Viewport ${slot + 1}`}
        </span>
        {!hideDropdown && (
          <select
            value={activeComponent}
            onChange={handleComponentChange}
            className="viewport-dropdown"
            disabled={!imageSrc}
          >
            <option value="spatial">Spatial</option>
            <option value="magnitude">FT Magnitude</option>
            <option value="phase">FT Phase</option>
            <option value="real">FT Real</option>
            <option value="imaginary">FT Imaginary</option>
          </select>
        )}
      </div>

      {/* Image area */}
      <div
        className={`viewport-canvas ${isDragging ? 'dragging' : ''} ${allowUpload ? 'clickable' : ''}`}
        onDoubleClick={handleDoubleClick}
        onMouseDown={handlers.onMouseDown}
        onMouseMove={handlers.onMouseMove}
        onMouseUp={handlers.onMouseUp}
        onMouseLeave={handlers.onMouseLeave}
      >
        {loading && (
          <div className="viewport-loading">
            <div className="spinner" />
          </div>
        )}

        {imageSrc ? (
          <img src={imageSrc} alt={filename || 'Viewport image'} className="viewport-image" draggable={false} />
        ) : (
          <div className="viewport-empty">
            <svg className="viewport-empty-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="viewport-empty-text">
              {allowUpload ? 'Double-click to load' : 'Waiting for input'}
            </span>
          </div>
        )}

        {/* Region overlay */}
        {regionOverlay && imageSrc && (
          <div className="region-overlay-container">
            {regionOverlay.type === 'outer' && (
              <div className="region-overlay-outer" />
            )}
            <div
              className={`region-overlay-rect ${regionOverlay.type}`}
              style={{
                left: `${(100 - regionOverlay.size) / 2}%`,
                top: `${(100 - regionOverlay.size) / 2}%`,
                width: `${regionOverlay.size}%`,
                height: `${regionOverlay.size}%`,
              }}
            />
          </div>
        )}

        {error && (
          <div className="viewport-error">{error}</div>
        )}
      </div>

      {/* Footer */}
      <div className="viewport-footer">
        <span className="viewport-filename">{filename || 'No image'}</span>
        <span>B:{brightness.toFixed(2)} C:{contrast.toFixed(2)}</span>
      </div>

      <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.bmp" onChange={handleFileChange} className="hidden" />
    </div>
  );
}
