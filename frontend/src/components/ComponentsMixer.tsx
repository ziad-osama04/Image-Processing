/**
 * ComponentsMixer.tsx — Mixer control panel for Part A.
 *
 * Provides:
 *  - Mix mode selection (Magnitude/Phase vs Real/Imaginary)
 *  - Weight sliders for each input image (4 images × 2 components)
 *  - Region selection (size slider + inner/outer toggle)
 *  - Mix button + simulate slow toggle
 *  - Progress bar with cancel
 */

import { useState, useCallback } from 'react';
import type { MixMode, ImageWeight } from '../types/image';

interface ComponentsMixerProps {
  onMix: (
    mode: MixMode,
    weights: ImageWeight[],
    regionSize: number,
    regionType: 'inner' | 'outer',
    simulateSlow: boolean
  ) => void;
  onCancel: () => void;
  progress: number | null; // null = not mixing, 0–100 = mixing
  loadedSlots: boolean[]; // which of the 4 input slots have images
}

export function ComponentsMixer({ onMix, onCancel, progress, loadedSlots }: ComponentsMixerProps) {
  const [mode, setMode] = useState<MixMode>('mag-phase');
  const [weights, setWeights] = useState<ImageWeight[]>([
    { componentA: 1, componentB: 0 },
    { componentA: 0, componentB: 1 },
    { componentA: 0, componentB: 0 },
    { componentA: 0, componentB: 0 },
  ]);
  const [regionSize, setRegionSize] = useState(100);
  const [regionType, setRegionType] = useState<'inner' | 'outer'>('inner');
  const [simulateSlow, setSimulateSlow] = useState(false);

  const labelA = mode === 'mag-phase' ? 'Magnitude' : 'Real';
  const labelB = mode === 'mag-phase' ? 'Phase' : 'Imaginary';
  const isMixing = progress !== null;
  const hasAnyImage = loadedSlots.some(Boolean);

  const updateWeight = useCallback((idx: number, key: 'componentA' | 'componentB', value: number) => {
    setWeights(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], [key]: value };
      return next;
    });
  }, []);

  const handleMix = () => {
    onMix(mode, weights, regionSize, regionType, simulateSlow);
  };

  return (
    <div className="mixer-panel">
      <div className="mixer-panel-header">
        <h3 className="mixer-panel-title">Components Mixer</h3>
        <div className="mixer-mode-switch">
          <button
            className={`mixer-mode-btn ${mode === 'mag-phase' ? 'active' : ''}`}
            onClick={() => setMode('mag-phase')}
          >
            Mag / Phase
          </button>
          <button
            className={`mixer-mode-btn ${mode === 'real-imag' ? 'active' : ''}`}
            onClick={() => setMode('real-imag')}
          >
            Real / Imag
          </button>
        </div>
      </div>

      {/* Weight Sliders */}
      <div className="mixer-weights">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className={`mixer-weight-row ${!loadedSlots[i] ? 'disabled' : ''}`}>
            <span className="mixer-weight-label">Image {i + 1}</span>
            <div className="mixer-slider-group">
              <label className="mixer-slider-label">{labelA}</label>
              <input
                type="range"
                min="0" max="1" step="0.01"
                value={weights[i].componentA}
                onChange={e => updateWeight(i, 'componentA', parseFloat(e.target.value))}
                disabled={!loadedSlots[i] || isMixing}
                className="mixer-slider"
              />
              <span className="mixer-slider-value">{weights[i].componentA.toFixed(2)}</span>
            </div>
            <div className="mixer-slider-group">
              <label className="mixer-slider-label">{labelB}</label>
              <input
                type="range"
                min="0" max="1" step="0.01"
                value={weights[i].componentB}
                onChange={e => updateWeight(i, 'componentB', parseFloat(e.target.value))}
                disabled={!loadedSlots[i] || isMixing}
                className="mixer-slider"
              />
              <span className="mixer-slider-value">{weights[i].componentB.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Region Controls */}
      <div className="mixer-region">
        <div className="mixer-region-header">
          <span className="mixer-region-title">Frequency Region</span>
          <div className="mixer-region-toggle">
            <button
              className={`region-btn ${regionType === 'inner' ? 'active' : ''}`}
              onClick={() => setRegionType('inner')}
              disabled={isMixing}
            >
              Inner (Low)
            </button>
            <button
              className={`region-btn ${regionType === 'outer' ? 'active' : ''}`}
              onClick={() => setRegionType('outer')}
              disabled={isMixing}
            >
              Outer (High)
            </button>
          </div>
        </div>
        <div className="mixer-slider-group">
          <label className="mixer-slider-label">Size</label>
          <input
            type="range"
            min="0" max="100" step="1"
            value={regionSize}
            onChange={e => setRegionSize(parseInt(e.target.value))}
            disabled={isMixing}
            className="mixer-slider region-slider"
          />
          <span className="mixer-slider-value">{regionSize}%</span>
        </div>
      </div>

      {/* Actions */}
      <div className="mixer-actions">
        <label className="mixer-simulate-label">
          <input
            type="checkbox"
            className="mixer-simulate-checkbox"
            checked={simulateSlow}
            onChange={e => setSimulateSlow(e.target.checked)}
            disabled={isMixing}
          />
          Simulate slow processing
        </label>

        {!isMixing ? (
          <button
            onClick={handleMix}
            disabled={!hasAnyImage}
            className="mixer-mix-btn"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Mix →
          </button>
        ) : (
          <button onClick={onCancel} className="mixer-cancel-btn">
            Cancel
          </button>
        )}
      </div>

      {/* Progress Bar */}
      {isMixing && (
        <div className="mixer-progress">
          <div className="mixer-progress-bar">
            <div className="mixer-progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="mixer-progress-text">{Math.round(progress!)}%</span>
        </div>
      )}
    </div>
  );
}
