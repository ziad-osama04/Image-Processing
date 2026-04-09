import type { EmphasizerAction, EmphasizerParams, WindowType } from '../types/image';

function ParamSlider({ label, value, min, max, step, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="param-slider">
      <label className="param-slider-label">{label}</label>
      <div className="param-slider-row">
        <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} className="mixer-slider" />
        <input type="number" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value) || 0)} className="param-number-input" />
      </div>
    </div>
  );
}

interface DynamicParameterPanelProps {
  action: EmphasizerAction;
  params: EmphasizerParams;
  onParamChange: <K extends keyof EmphasizerParams>(key: K, value: EmphasizerParams[K]) => void;
}

export function DynamicParameterPanel({ action, params, onParamChange }: DynamicParameterPanelProps) {
  return (
    <div className="action-params">
      {action === 'shift' && (
        <>
          <ParamSlider label="Shift X" value={params.shiftX} min={-100} max={100} step={1} onChange={v => onParamChange('shiftX', v)} />
          <ParamSlider label="Shift Y" value={params.shiftY} min={-100} max={100} step={1} onChange={v => onParamChange('shiftY', v)} />
        </>
      )}
      {action === 'complex-exponential' && (
        <>
          <ParamSlider label="Frequency U" value={params.expU} min={-50} max={50} step={0.5} onChange={v => onParamChange('expU', v)} />
          <ParamSlider label="Frequency V" value={params.expV} min={-50} max={50} step={0.5} onChange={v => onParamChange('expV', v)} />
        </>
      )}
      {action === 'stretch' && (
        <ParamSlider label="Factor" value={params.stretchFactor} min={0.1} max={4} step={0.1} onChange={v => onParamChange('stretchFactor', v)} />
      )}
      {action === 'mirror' && (
        <div className="action-panel-section">
          <label className="action-label">Axis</label>
          <select value={params.mirrorAxis} onChange={e => onParamChange('mirrorAxis', e.target.value as 'horizontal' | 'vertical' | 'both')} className="action-select">
            <option value="horizontal">Horizontal</option>
            <option value="vertical">Vertical</option>
            <option value="both">Both</option>
          </select>
        </div>
      )}
      {action === 'even-odd' && (
        <div className="action-panel-section">
          <label className="action-label">Type</label>
          <div className="domain-toggle">
            <button className={`domain-btn ${params.evenOddType === 'even' ? 'active' : ''}`} onClick={() => onParamChange('evenOddType', 'even')}>Even</button>
            <button className={`domain-btn ${params.evenOddType === 'odd' ? 'active' : ''}`} onClick={() => onParamChange('evenOddType', 'odd')}>Odd</button>
          </div>
        </div>
      )}
      {action === 'rotate' && (
        <ParamSlider label="Angle (°)" value={params.rotateAngle} min={0} max={360} step={1} onChange={v => onParamChange('rotateAngle', v)} />
      )}
      {action === 'differentiate' && (
        <div className="action-panel-section">
          <label className="action-label">Direction</label>
          <select value={params.diffDirection} onChange={e => onParamChange('diffDirection', e.target.value as 'x' | 'y' | 'both')} className="action-select">
            <option value="x">Horizontal (X)</option>
            <option value="y">Vertical (Y)</option>
            <option value="both">Both (Gradient)</option>
          </select>
        </div>
      )}
      {action === 'integrate' && (
        <div className="action-panel-section">
          <label className="action-label">Direction</label>
          <select value={params.intDirection} onChange={e => onParamChange('intDirection', e.target.value as 'x' | 'y' | 'both')} className="action-select">
            <option value="x">Horizontal (X)</option>
            <option value="y">Vertical (Y)</option>
            <option value="both">Both</option>
          </select>
        </div>
      )}
      {action === 'window' && (
        <>
          <div className="action-panel-section">
            <label className="action-label">Window Type</label>
            <select value={params.windowType} onChange={e => onParamChange('windowType', e.target.value as WindowType)} className="action-select">
              <option value="rectangular">Rectangular</option>
              <option value="gaussian">Gaussian</option>
              <option value="hamming">Hamming</option>
              <option value="hanning">Hanning</option>
            </select>
          </div>
          <ParamSlider label="Kernel Width" value={params.windowKernelWidth} min={3} max={51} step={2} onChange={v => onParamChange('windowKernelWidth', Math.round(v) | 1)} />
          <ParamSlider label="Kernel Height" value={params.windowKernelHeight} min={3} max={51} step={2} onChange={v => onParamChange('windowKernelHeight', Math.round(v) | 1)} />
          <ParamSlider label="Stride X" value={params.windowStrideX} min={1} max={10} step={1} onChange={v => onParamChange('windowStrideX', v)} />
          <ParamSlider label="Stride Y" value={params.windowStrideY} min={1} max={10} step={1} onChange={v => onParamChange('windowStrideY', v)} />
          <div className="action-panel-section">
            <label className="action-label">Conv Mode</label>
            <select value={params.windowMode} onChange={e => onParamChange('windowMode', e.target.value as 'same' | 'valid')} className="action-select">
              <option value="same">Same (preserve size)</option>
              <option value="valid">Valid (no padding)</option>
            </select>
          </div>
          {params.windowType === 'gaussian' && (
            <ParamSlider label="Sigma" value={params.windowSigma} min={0.1} max={5} step={0.1} onChange={v => onParamChange('windowSigma', v)} />
          )}
        </>
      )}
    </div>
  );
}
