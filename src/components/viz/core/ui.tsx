import type { ReactNode } from 'react';
import Algo, { type AlgoMood } from '@/components/mascot/Algo';

/* Shared controls for visualizations. Styles live in src/styles/viz.css (vz-*). */

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format = (v: number) => String(v),
  accent,
  disabled,
  hint
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  /** CSS color for the filled track. */
  accent?: string;
  disabled?: boolean;
  hint?: ReactNode;
}) {
  const t = max === min ? 0 : ((value - min) / (max - min)) * 100;
  return (
    <label className={`vz-slider${disabled ? ' is-disabled' : ''}`}>
      <span className="vz-slider-top">
        <span className="vz-label">{label}</span>
        <output className="vz-value">{format(value)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--t' as string]: `${t}%`, ...(accent ? { ['--accent' as string]: accent } : {}) }}
      />
      {hint && <span className="vz-hint">{hint}</span>}
    </label>
  );
}

export function Button({
  children,
  onClick,
  variant = 'default',
  active = false,
  disabled = false,
  title,
  size = 'md'
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'primary' | 'ghost' | 'danger';
  active?: boolean;
  disabled?: boolean;
  title?: string;
  size?: 'sm' | 'md';
}) {
  return (
    <button
      type="button"
      className={`vz-btn vz-btn-${variant} vz-btn-${size}${active ? ' is-active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-pressed={variant === 'default' ? active : undefined}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label
}: {
  options: { value: T; label: ReactNode; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  label?: ReactNode;
}) {
  return (
    <div className="vz-seg-wrap">
      {label && <span className="vz-label">{label}</span>}
      <div className="vz-seg" role="radiogroup">
        {options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            title={o.title}
            className={o.value === value ? 'is-active' : ''}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="vz-toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="vz-toggle-track" aria-hidden="true">
        <span className="vz-toggle-thumb" />
      </span>
      <span className="vz-label">{label}</span>
    </label>
  );
}

export function Stat({
  label,
  value,
  tone,
  sub
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: 'good' | 'bad' | 'warn' | 'brand';
  sub?: ReactNode;
}) {
  return (
    <div className={`vz-stat${tone ? ` vz-stat-${tone}` : ''}`}>
      <span className="vz-stat-label">{label}</span>
      <span className="vz-stat-value">{value}</span>
      {sub && <span className="vz-stat-sub">{sub}</span>}
    </div>
  );
}

export function Controls({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`vz-controls ${className}`}>{children}</div>;
}

export function Panel({ title, children }: { title?: ReactNode; children: ReactNode }) {
  return (
    <div className="vz-panel">
      {title && <div className="vz-panel-title">{title}</div>}
      {children}
    </div>
  );
}

export function Legend({ items }: { items: { color: string; label: ReactNode; dashed?: boolean }[] }) {
  return (
    <div className="vz-legend">
      {items.map((it, i) => (
        <span key={i} className="vz-legend-item">
          <span
            className="vz-legend-swatch"
            style={{ background: it.dashed ? 'transparent' : it.color, borderColor: it.color, borderStyle: it.dashed ? 'dashed' : 'solid' }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Play / pause / step / reset bar for step-based simulations. */
export function PlayBar({
  playing,
  onToggle,
  onStep,
  onReset,
  speed,
  onSpeed,
  extra
}: {
  playing: boolean;
  onToggle: () => void;
  onStep?: () => void;
  onReset?: () => void;
  speed?: number;
  onSpeed?: (v: number) => void;
  extra?: ReactNode;
}) {
  return (
    <div className="vz-playbar">
      <Button variant="primary" onClick={onToggle}>
        {playing ? '❚❚ 멈춤' : '▶ 시작'}
      </Button>
      {onStep && (
        <Button onClick={onStep} disabled={playing} title="한 걸음씩">
          한 걸음 ⟶
        </Button>
      )}
      {onReset && (
        <Button variant="ghost" onClick={onReset} title="처음으로">
          ↺ 처음으로
        </Button>
      )}
      {onSpeed && speed !== undefined && (
        <Segmented
          options={[
            { value: 0.5, label: '느리게' },
            { value: 1, label: '보통' },
            { value: 3, label: '빠르게' }
          ]}
          value={speed}
          onChange={onSpeed}
        />
      )}
      {extra}
    </div>
  );
}

/** A small 알고 note inside a visualization (results, warnings, hints). */
export function Note({ children, mood = 'smile', tone }: { children: ReactNode; mood?: AlgoMood; tone?: 'good' | 'bad' | 'warn' }) {
  return (
    <div className={`vz-note${tone ? ` vz-note-${tone}` : ''}`} role="status">
      <Algo mood={mood} size={34} still />
      <div className="vz-note-text">{children}</div>
    </div>
  );
}

/**
 * Responsive two-area layout: a stage (canvas / svg) and a side panel.
 * Side-by-side on wide frames, stacked on narrow ones.
 */
export function VizLayout({ stage, side, sideWidth = 300 }: { stage: ReactNode; side?: ReactNode; sideWidth?: number }) {
  return (
    <div className="vz-layout" style={{ ['--side-w' as string]: `${sideWidth}px` }}>
      <div className="vz-stage">{stage}</div>
      {side && <div className="vz-side">{side}</div>}
    </div>
  );
}
