/**
 * ProgressBar.tsx — Animated progress bar for transform operations.
 *
 * Shows a gradient-filled progress bar with percentage text.
 * Hides when `visible` is false with a smooth transition.
 */

interface ProgressBarProps {
  /** Progress value 0.0–1.0 */
  progress: number;
  /** Whether the progress bar is visible */
  visible: boolean;
}

export function ProgressBar({ progress, visible }: ProgressBarProps) {
  const percent = Math.round(Math.max(0, Math.min(1, progress)) * 100);

  return (
    <div
      className="progress-bar-container"
      style={{
        opacity: visible ? 1 : 0,
        maxHeight: visible ? '40px' : '0px',
        overflow: 'hidden',
        transition: 'opacity 0.3s ease, max-height 0.3s ease',
      }}
    >
      <div className="progress-bar-track">
        <div
          className={`progress-bar-fill ${visible && percent > 0 && percent < 100 ? 'pulse' : ''}`}
          style={{ width: `${percent}%`, transition: 'width 0.3s ease-out' }}
        />
      </div>
      <span className="progress-bar-text">{percent}%</span>
    </div>
  );
}
