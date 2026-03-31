/**
 * DomainIndicator.tsx — Spatial / Frequency domain toggle.
 *
 * Extracted from EmphasizerWorkspace for reusability (Constitution §III).
 */

interface DomainIndicatorProps {
  applyInFrequency: boolean;
  onChange: (applyInFrequency: boolean) => void;
}

export function DomainIndicator({ applyInFrequency, onChange }: DomainIndicatorProps) {
  return (
    <div className="action-panel-section">
      <label className="action-label">Apply in Domain</label>
      <div className="domain-toggle">
        <button
          className={`domain-btn ${!applyInFrequency ? 'active' : ''}`}
          onClick={() => onChange(false)}
          style={!applyInFrequency ? { backgroundColor: '#3b82f6', color: 'white' } : {}}
        >
          Spatial
        </button>
        <button
          className={`domain-btn ${applyInFrequency ? 'active' : ''}`}
          onClick={() => onChange(true)}
          style={applyInFrequency ? { backgroundColor: '#10b981', color: 'white' } : {}}
        >
          Frequency
        </button>
      </div>
    </div>
  );
}
