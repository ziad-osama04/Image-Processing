import type { EmphasizerAction } from '../types/image';

const ACTION_LABELS: Record<EmphasizerAction, string> = {
  'shift': 'Shift Image',
  'complex-exponential': 'Complex Exponential',
  'stretch': 'Stretch / Scale',
  'mirror': 'Mirror (Symmetry)',
  'even-odd': 'Make Even / Odd',
  'rotate': 'Rotate',
  'differentiate': 'Differentiate',
  'integrate': 'Integrate',
  'window': '2D Window',
};

interface OperationSelectorProps {
  action: EmphasizerAction;
  onChange: (action: EmphasizerAction) => void;
}

export function OperationSelector({ action, onChange }: OperationSelectorProps) {
  return (
    <div className="action-panel-section">
      <label className="action-label">Action</label>
      <select
        value={action}
        onChange={e => onChange(e.target.value as EmphasizerAction)}
        className="action-select"
      >
        {Object.entries(ACTION_LABELS).map(([key, label]) => (
          <option key={key} value={key}>{label}</option>
        ))}
      </select>
    </div>
  );
}
