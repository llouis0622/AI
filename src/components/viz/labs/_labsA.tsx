/**
 * Small SVG drawing kit for the code-lab walkthroughs (labs 01–10).
 * Scenes use a 480×300 viewBox so labels stay readable on phones.
 */
import type { ReactNode } from 'react';
import { withAlpha, type VizColors } from '../core/colors';

/** Progress of a sub-animation that runs while t goes from a to b (clamped 0..1). */
export const seg = (t: number, a: number, b: number) => Math.min(1, Math.max(0, (t - a) / (b - a)));
export const easeIO = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;

const hex = (s: string) => (s.startsWith('#') ? s : '#7c6cf6');
export const soft = (color: string, a = 0.16) => withAlpha(hex(color), a);

export const VB: [number, number] = [480, 300];

/** Keyframes for flowing dashes / pulses (scoped names, reduced-motion aware). */
export function Defs() {
  return (
    <style>{`
      .labA-flow { stroke-dasharray: 7 6; animation: labA-dash 1.1s linear infinite; }
      .labA-pulse { animation: labA-pulse 1.6s ease-in-out infinite; }
      .labA-bob { animation: labA-bob 2.4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
      @keyframes labA-dash { to { stroke-dashoffset: -26; } }
      @keyframes labA-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
      @keyframes labA-bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
      @media (prefers-reduced-motion: reduce) { .labA-flow, .labA-pulse, .labA-bob { animation: none; } }
    `}</style>
  );
}

export function Txt({
  x,
  y,
  children,
  size = 15,
  weight = 700,
  color = 'var(--text)',
  anchor = 'middle',
  mono = false,
  opacity = 1
}: {
  x: number;
  y: number;
  children: ReactNode;
  size?: number;
  weight?: number;
  color?: string;
  anchor?: 'start' | 'middle' | 'end';
  mono?: boolean;
  opacity?: number;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      opacity={opacity}
      style={{ fontSize: size, fontWeight: weight, fill: color, fontFamily: mono ? 'var(--font-mono)' : 'var(--font-sans)' }}
    >
      {children}
    </text>
  );
}

/** Labelled rounded box (a module, a tensor, a stage). */
export function Box({
  x,
  y,
  w,
  h,
  label,
  sub,
  color,
  t = 1,
  fs = 16,
  subFs = 12.5,
  dim = false,
  glow = false,
  dashed = false,
  c
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: ReactNode;
  sub?: ReactNode;
  color: string;
  t?: number;
  fs?: number;
  subFs?: number;
  dim?: boolean;
  glow?: boolean;
  dashed?: boolean;
  c: VizColors;
}) {
  const s = 0.82 + 0.18 * t;
  const dark = c.theme === 'dark';
  return (
    <g
      opacity={Math.min(1, t * 1.8) * (dim ? 0.38 : 1)}
      transform={`translate(${x + w / 2} ${y + h / 2}) scale(${s}) translate(${-w / 2} ${-h / 2})`}
    >
      {glow && <rect x={-5} y={-5} width={w + 10} height={h + 10} rx={16} fill="none" stroke={color} strokeWidth={2} opacity={0.35} className="labA-pulse" />}
      <rect
        width={w}
        height={h}
        rx={12}
        fill={soft(color, dark ? 0.2 : 0.15)}
        stroke={color}
        strokeWidth={2}
        strokeDasharray={dashed ? '6 5' : undefined}
      />
      {label !== undefined && (
        <Txt x={w / 2} y={sub ? h / 2 - 2 : h / 2 + fs * 0.35} size={fs} weight={800}>
          {label}
        </Txt>
      )}
      {sub !== undefined && (
        <Txt x={w / 2} y={h / 2 + subFs + 3} size={subFs} weight={650} color="var(--text-2)" mono>
          {sub}
        </Txt>
      )}
    </g>
  );
}

/** A stacked "batch" tensor: a few offset cards with a shape caption. */
export function Tensor({
  x,
  y,
  w,
  h,
  label,
  shape,
  color,
  t = 1,
  depth = 3,
  c
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: ReactNode;
  shape?: string;
  color: string;
  t?: number;
  depth?: number;
  c: VizColors;
}) {
  const dark = c.theme === 'dark';
  const cards = [];
  for (let k = depth - 1; k >= 0; k--) {
    const off = k * 5 * t;
    cards.push(
      <rect
        key={k}
        x={x + off}
        y={y - off}
        width={w}
        height={h}
        rx={9}
        fill={k === 0 ? soft(color, dark ? 0.24 : 0.18) : soft(color, dark ? 0.12 : 0.08)}
        stroke={color}
        strokeWidth={k === 0 ? 2 : 1.2}
        opacity={k === 0 ? 1 : 0.7}
      />
    );
  }
  return (
    <g opacity={Math.min(1, t * 1.8)}>
      {cards}
      <Txt x={x + w / 2} y={y + h / 2 + 6} size={17} weight={850}>
        {label}
      </Txt>
      {shape && (
        <Txt x={x + w / 2} y={y + h + 18} size={12.5} weight={650} color="var(--text-2)" mono>
          {shape}
        </Txt>
      )}
    </g>
  );
}

/** Self-drawing arrow (straight, or curved with `bend`). After it is drawn it can keep flowing. */
export function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  t = 1,
  width = 2.5,
  bend = 0,
  flow = false,
  label,
  labelDy = -8,
  labelSize = 12.5,
  head = true
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  t?: number;
  width?: number;
  bend?: number;
  flow?: boolean;
  label?: ReactNode;
  labelDy?: number;
  labelSize?: number;
  head?: boolean;
}) {
  if (t <= 0) return null;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const nx = -(y2 - y1) / len;
  const ny = (x2 - x1) / len;
  const cx = mx + nx * bend;
  const cy = my + ny * bend;
  const at = (s: number): [number, number] => [
    (1 - s) ** 2 * x1 + 2 * (1 - s) * s * cx + s * s * x2,
    (1 - s) ** 2 * y1 + 2 * (1 - s) * s * cy + s * s * y2
  ];
  const tan = (s: number): [number, number] => [2 * (1 - s) * (cx - x1) + 2 * s * (x2 - cx), 2 * (1 - s) * (cy - y1) + 2 * s * (y2 - cy)];
  const [ex, ey] = at(t);
  const [tx, ty] = tan(Math.max(0.02, t));
  const ang = Math.atan2(ty, tx);
  const hl = 9 + width;
  const d = `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`;
  const done = t >= 0.999;
  return (
    <g>
      {done ? (
        <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" className={flow ? 'labA-flow' : undefined} />
      ) : (
        <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray={`${t} 1`} />
      )}
      {head && (
        <path
          d={`M${ex} ${ey} L${ex - hl * Math.cos(ang - 0.42)} ${ey - hl * Math.sin(ang - 0.42)} L${ex - hl * Math.cos(ang + 0.42)} ${ey - hl * Math.sin(ang + 0.42)} Z`}
          fill={color}
        />
      )}
      {label !== undefined && t > 0.55 && (
        <Txt x={cx} y={cy + labelDy} size={labelSize} weight={750} color="var(--text-2)">
          {label}
        </Txt>
      )}
    </g>
  );
}

/** Token chip centered on (x, y). */
export function Chip({
  x,
  y,
  text,
  color,
  w,
  h = 28,
  fs = 14,
  t = 1,
  mono = false,
  active = false,
  dim = false,
  c
}: {
  x: number;
  y: number;
  text: ReactNode;
  color: string;
  w?: number;
  h?: number;
  fs?: number;
  t?: number;
  mono?: boolean;
  active?: boolean;
  dim?: boolean;
  c: VizColors;
}) {
  const width = w ?? 34;
  const dark = c.theme === 'dark';
  return (
    <g opacity={Math.min(1, t * 2) * (dim ? 0.3 : 1)} transform={`translate(${x} ${y}) scale(${0.7 + 0.3 * Math.min(1, t * 1.4)})`}>
      <rect
        x={-width / 2}
        y={-h / 2}
        width={width}
        height={h}
        rx={8}
        fill={active ? color : soft(color, dark ? 0.24 : 0.17)}
        stroke={color}
        strokeWidth={active ? 0 : 1.5}
      />
      <Txt x={0} y={fs * 0.36} size={fs} weight={800} mono={mono} color={active ? (dark ? '#041018' : '#ffffff') : 'var(--text)'}>
        {text}
      </Txt>
    </g>
  );
}

/** Vertical bars that grow with t. */
export function Bars({
  x,
  y,
  w,
  h,
  values,
  labels,
  color,
  highlight = -1,
  t = 1,
  max,
  fs = 12,
  hiColor,
  valueText
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  values: number[];
  labels?: ReactNode[];
  color: string;
  highlight?: number;
  t?: number;
  max?: number;
  fs?: number;
  hiColor?: string;
  valueText?: (v: number, i: number) => ReactNode;
}) {
  const m = max ?? Math.max(...values, 1e-9);
  const n = values.length;
  const gap = Math.min(8, (w / n) * 0.25);
  const bw = (w - gap * (n - 1)) / n;
  return (
    <g>
      <line x1={x - 2} x2={x + w + 2} y1={y + h} y2={y + h} stroke="var(--viz-axis)" strokeWidth={1.5} />
      {values.map((v, i) => {
        const bh = (Math.max(0, v) / m) * h * t;
        const bx = x + i * (bw + gap);
        const hi = i === highlight;
        return (
          <g key={i}>
            <rect x={bx} y={y + h - bh} width={bw} height={bh} rx={Math.min(5, bw / 3)} fill={hi ? hiColor ?? color : soft(color, 0.55)} />
            {labels?.[i] !== undefined && (
              <Txt x={bx + bw / 2} y={y + h + fs + 4} size={fs} weight={hi ? 850 : 650} color={hi ? 'var(--text)' : 'var(--text-2)'}>
                {labels[i]}
              </Txt>
            )}
            {valueText && t > 0.7 && (
              <Txt x={bx + bw / 2} y={y + h - bh - 6} size={fs} weight={800} color={hi ? 'var(--text)' : 'var(--text-2)'}>
                {valueText(v, i)}
              </Txt>
            )}
          </g>
        );
      })}
    </g>
  );
}

/** Small line plot of fn over xDomain that draws itself as t goes 0 → 1. */
export function Curve({
  x,
  y,
  w,
  h,
  fn,
  xDomain,
  yDomain,
  t = 1,
  color,
  n = 90,
  xLabel,
  yLabel,
  marker,
  markerLabel,
  markerAnchor = 'start',
  width = 3
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fn: (v: number) => number;
  xDomain: [number, number];
  yDomain: [number, number];
  t?: number;
  color: string;
  n?: number;
  xLabel?: ReactNode;
  yLabel?: ReactNode;
  marker?: number;
  markerLabel?: ReactNode;
  markerAnchor?: 'start' | 'end';
  width?: number;
}) {
  const sx = (v: number) => x + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * w;
  const sy = (v: number) => y + h - ((Math.min(yDomain[1], Math.max(yDomain[0], v)) - yDomain[0]) / (yDomain[1] - yDomain[0])) * h;
  const last = Math.max(1, Math.round(n * t));
  const pts: string[] = [];
  for (let k = 0; k <= last; k++) {
    const v = xDomain[0] + ((xDomain[1] - xDomain[0]) * k) / n;
    pts.push(`${sx(v).toFixed(1)},${sy(fn(v)).toFixed(1)}`);
  }
  return (
    <g>
      <line x1={x} x2={x + w} y1={y + h} y2={y + h} stroke="var(--viz-axis)" strokeWidth={1.5} />
      <line x1={x} x2={x} y1={y} y2={y + h} stroke="var(--viz-axis)" strokeWidth={1.5} />
      {xLabel && (
        <Txt x={x + w / 2} y={y + h + 20} size={12.5} weight={700} color="var(--text-2)">
          {xLabel}
        </Txt>
      )}
      {yLabel && (
        <Txt x={x - 8} y={y - 8} size={12.5} weight={700} color="var(--text-2)" anchor="start">
          {yLabel}
        </Txt>
      )}
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={width} strokeLinejoin="round" strokeLinecap="round" />
      {marker !== undefined && t > 0.85 && (
        <g>
          <line x1={sx(marker)} x2={sx(marker)} y1={sy(fn(marker))} y2={y + h} stroke={color} strokeDasharray="3 4" strokeWidth={1.5} />
          <circle cx={sx(marker)} cy={sy(fn(marker))} r={6} fill={color} stroke="var(--surface-solid)" strokeWidth={2} />
          {markerLabel && (
            <Txt x={sx(marker) + (markerAnchor === 'start' ? 9 : -9)} y={sy(fn(marker)) - 9} size={12.5} weight={800} anchor={markerAnchor}>
              {markerLabel}
            </Txt>
          )}
        </g>
      )}
    </g>
  );
}

/** A matrix of square cells; `value(r, k)` ∈ [0,1] tints, `hidden(r, k)` greys out. */
export function Cells({
  x,
  y,
  rows,
  cols,
  cell = 14,
  gap = 2,
  color,
  t = 1,
  value,
  hidden,
  order = 'row'
}: {
  x: number;
  y: number;
  rows: number;
  cols: number;
  cell?: number;
  gap?: number;
  color: string;
  t?: number;
  value?: (r: number, k: number) => number;
  hidden?: (r: number, k: number) => boolean;
  order?: 'row' | 'diag';
}) {
  const out: ReactNode[] = [];
  const total = rows * cols;
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < cols; k++) {
      const rank = order === 'diag' ? (r + k) / Math.max(1, rows + cols - 2) : (r * cols + k) / Math.max(1, total - 1);
      const appear = Math.min(1, Math.max(0, (t * 1.25 - rank) * 4));
      if (appear <= 0) continue;
      const off = hidden?.(r, k);
      const v = value?.(r, k) ?? 0.4;
      out.push(
        <rect
          key={`${r}-${k}`}
          x={x + k * (cell + gap)}
          y={y + r * (cell + gap)}
          width={cell}
          height={cell}
          rx={Math.min(4, cell / 4)}
          opacity={appear}
          fill={off ? 'var(--surface-3)' : soft(color, 0.12 + 0.82 * v)}
          stroke={off ? 'var(--border-strong)' : 'none'}
          strokeWidth={off ? 1 : 0}
        />
      );
    }
  return <g>{out}</g>;
}

/** Emoji glyph as SVG text. */
export function Emoji({ x, y, children, size = 26, className }: { x: number; y: number; children: ReactNode; size?: number; className?: string }) {
  return (
    <text x={x} y={y} textAnchor="middle" className={className} style={{ fontSize: size }}>
      {children}
    </text>
  );
}
