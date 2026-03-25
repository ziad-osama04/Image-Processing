/**
 * ViewportComponent: Reusable image/FT viewport.
 *
 * Single shared component for ALL viewports in the app (Constitution §III).
 * Supports: image display, FT component dropdown, mouse-drag B/C,
 * double-click browse, and region overlay (Phase 3).
 */

import { useCallback, useRef, useState } from 'react';
import type { FTComponent } from '../types/image';
import { getImage, getFTComponent, uploadImage } from '../services/imageApi';
import { useMouseDrag } from '../hooks/useMouseDrag';

interface ViewportComponentProps {
  sessionId: string;
  slot: number;
  label?: string;
  allowUpload?: boolean;
  onImageLoaded?: (slot: number, filename: string) => void;
}

export function ViewportComponent({
  sessionId,
  slot,
  label,
  allowUpload = true,
  onImageLoaded,
}: ViewportComponentProps) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [filename, setFilename] = useState('');
  const [activeComponent, setActiveComponent] = useState<'spatial' | FTComponent>('spatial');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { brightness, contrast, isDragging, handlers, reset } = useMouseDrag();

  // Fetch the image with current B/C settings
  const fetchDisplay = useCallback(
    async (component: 'spatial' | FTComponent, b: number, c: number) => {
      if (!sessionId) return;
      setLoading(true);
      setError(null);
      try {
        if (component === 'spatial') {
          const res = await getImage(sessionId, slot, b, c);
          setImageSrc(`data:image/png;base64,${res.preview}`);
          setFilename(res.filename);
        } else {
          const res = await getFTComponent(sessionId, slot, component, b, c);
          setImageSrc(`data:image/png;base64,${res.image}`);
        }
      } catch (err: unknown) {
        if (err instanceof Error && !err.message.includes('empty')) {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    },
    [sessionId, slot]
  );

  // Handle component dropdown change
  const handleComponentChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as 'spatial' | FTComponent;
      setActiveComponent(value);
      fetchDisplay(value, brightness, contrast);
    },
    [fetchDisplay, brightness, contrast]
  );

  // Handle double-click browse
  const handleDoubleClick = useCallback(() => {
    if (!allowUpload) return;
    fileInputRef.current?.click();
  }, [allowUpload]);

  // Handle file selection
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
        setActiveComponent('spatial');
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

  // Refresh display when B/C changes (via mouse drag)
  const handleMouseUp = useCallback(() => {
    handlers.onMouseUp();
    if (imageSrc) {
      fetchDisplay(activeComponent, brightness, contrast);
    }
  }, [handlers, imageSrc, fetchDisplay, activeComponent, brightness, contrast]);

  return (
    <div className="flex flex-col gap-2 bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl overflow-hidden shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[var(--bg-secondary)] border-b border-[var(--border-color)]">
        <span className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
          {label || `Viewport ${slot + 1}`}
        </span>
        <select
          value={activeComponent}
          onChange={handleComponentChange}
          className="text-xs bg-[var(--bg-primary)] text-[var(--text-primary)] border border-[var(--border-color)] rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]"
          disabled={!imageSrc}
        >
          <option value="spatial">Spatial</option>
          <option value="magnitude">FT Magnitude</option>
          <option value="phase">FT Phase</option>
          <option value="real">FT Real</option>
          <option value="imaginary">FT Imaginary</option>
        </select>
      </div>

      {/* Image area */}
      <div
        className={`relative w-full aspect-square flex items-center justify-center cursor-${
          isDragging ? 'grabbing' : allowUpload ? 'pointer' : 'default'
        } select-none`}
        onDoubleClick={handleDoubleClick}
        onMouseDown={handlers.onMouseDown}
        onMouseMove={handlers.onMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handlers.onMouseLeave}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
            <div className="w-6 h-6 border-2 border-[var(--accent-blue)] border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {imageSrc ? (
          <img
            src={imageSrc}
            alt={filename || 'Viewport image'}
            className="max-w-full max-h-full object-contain"
            draggable={false}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-[var(--text-secondary)]">
            <svg className="w-8 h-8 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="text-xs">
              {allowUpload ? 'Double-click to load' : 'Output'}
            </span>
          </div>
        )}

        {error && (
          <div className="absolute bottom-2 left-2 right-2 text-xs text-red-400 bg-red-900/50 rounded px-2 py-1">
            {error}
          </div>
        )}
      </div>

      {/* Footer info */}
      <div className="px-3 py-1.5 text-[10px] text-[var(--text-secondary)] flex justify-between border-t border-[var(--border-color)]">
        <span>{filename || 'No image'}</span>
        <span>
          B:{brightness.toFixed(2)} C:{contrast.toFixed(2)}
        </span>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.bmp"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
