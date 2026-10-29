import type { ReactNode } from 'react';
import { withAlpha } from '../core/colors';
import { clamp } from '../core/math';

/*
 * Drawing kit for the code-lab walkthroughs of labs 11–20 (Worker F).
 * Scenes are drawn in a 420-wide viewBox so text stays ≥ 11px on a 390px phone.
 */

export const VB: [number, number] = [420, 300];
/** Font sizes in viewBox units. */
export const FS = { h: 16, l: 14.5, s: 13, m: 12.5 };

/** Progress of a sub-animation that starts at `a` and lasts `d` (t units, 0..1). */
export const seg = (t: number, a: number, d = 0.35) => clamp((t - a) / d, 0, 1);

/** withAlpha needs hex; theme series colors are hex, fall back if not. */
export const hx = (s: string) => (s.startsWith('#') ? s : '#7c6cf6');
export const alpha = (color: string, a: number) => withAlpha(hx(color), a);

export const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Step text helper: **bold** and `code` markers → JSX. Avoids JSX's habit of
 * swallowing the space at a line break between text and an element.
 */
export function rich(s: string): ReactNode {
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`)/).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>;
    return part;
  });
}

/** Remove the common leading indentation of a code excerpt (keeps it verbatim otherwise). */
export function dd(code: string): string {
  const lines = code.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
  const indents = lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length);
  const cut = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.slice(cut)).join('\n');
}

/** Rough text width for layout (Hangul ≈ 1em, latin ≈ 0.58em). */
export function textWidth(s: string, size: number) {
  let w = 0;
  for (const ch of s) {
    if (/[ㄱ-힝]/.test(ch)) w += size * 0.98;
    else if (ch === ' ') w += size * 0.3;
    else if (/[A-Z0-9]/.test(ch)) w += size * 0.62;
    else w += size * 0.55;
  }
  return w;
}

export function T({
  x,
  y,
  children,
  size = FS.l,
  weight = 700,
  color = 'var(--text)',
  anchor = 'start',
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
      style={{ fontSize: size, fontWeight: weight, fill: color, fontFamily: mono ? 'var(--font-mono)' : undefined }}
    >
      {children}
    </text>
  );
}

/** Rounded box with a label (and optional mono sub-label). Pops in as t goes 0 → 1. */
export function Box({
  x,
  y,
  w,
  h,
  label,
  sub,
  color,
  t = 1,
  dashed = false,
  r = 12,
  fill = 0.16,
  labelSize = FS.l,
  solid = false
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: ReactNode;
  sub?: ReactNode;
  color: string;
  t?: number;
  dashed?: boolean;
  r?: number;
  fill?: number;
  labelSize?: number;
  solid?: boolean;
}) {
  const s = 0.86 + 0.14 * t;
  return (
    <g opacity={Math.min(1, t * 1.7)} transform={`translate(${x + w / 2} ${y + h / 2}) scale(${s}) translate(${-w / 2} ${-h / 2})`}>
      <rect
        width={w}
        height={h}
        rx={r}
        fill={solid ? color : alpha(color, fill)}
        stroke={color}
        strokeWidth={2}
        strokeDasharray={dashed ? '6 5' : undefined}
      />
      {label !== undefined && (
        <text
          x={w / 2}
          y={sub ? h / 2 - 3 : h / 2 + labelSize * 0.36}
          textAnchor="middle"
          style={{ fontSize: labelSize, fontWeight: 800, fill: solid ? '#fff' : 'var(--text)' }}
        >
          {label}
        </text>
      )}
      {sub && (
        <text x={w / 2} y={h / 2 + 15} textAnchor="middle" style={{ fontSize: FS.m, fontWeight: 650, fill: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
          {sub}
        </text>
      )}
    </g>
  );
}

/** Arrow that draws itself from (x1, y1) to (x2, y2) as t goes 0 → 1. */
export function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  t = 1,
  label,
  dashed = false,
  width = 2.5,
  labelDy = -7
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  t?: number;
  label?: ReactNode;
  dashed?: boolean;
  width?: number;
  labelDy?: number;
}) {
  if (t <= 0) return null;
  const xe = x1 + (x2 - x1) * t;
  const ye = y1 + (y2 - y1) * t;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const hl = 8.5;
  return (
    <g>
      <line x1={x1} y1={y1} x2={xe} y2={ye} stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dashed ? '5 5' : undefined} />
      {t > 0.08 && (
        <path
          d={`M${xe} ${ye} L${xe - hl * Math.cos(ang - 0.45)} ${ye - hl * Math.sin(ang - 0.45)} L${xe - hl * Math.cos(ang + 0.45)} ${ye - hl * Math.sin(ang + 0.45)} Z`}
          fill={color}
        />
      )}
      {label && t > 0.6 && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 + labelDy} textAnchor="middle" style={{ fontSize: FS.m, fontWeight: 750, fill: 'var(--text-2)' }}>
          {label}
        </text>
      )}
    </g>
  );
}

/** Pill-shaped tag centred at (x, y). */
export function Pill({
  x,
  y,
  label,
  color,
  t = 1,
  size = FS.s,
  solid = false,
  mono = false
}: {
  x: number;
  y: number;
  label: string;
  color: string;
  t?: number;
  size?: number;
  solid?: boolean;
  mono?: boolean;
}) {
  const w = textWidth(label, size) + 18;
  const h = size + 12;
  return (
    <g opacity={Math.min(1, t * 1.8)} transform={`translate(${x} ${y}) scale(${0.8 + 0.2 * t})`}>
      <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={h / 2} fill={solid ? color : alpha(color, 0.18)} stroke={color} strokeWidth={1.5} />
      <text
        x={0}
        y={size * 0.36}
        textAnchor="middle"
        style={{ fontSize: size, fontWeight: 800, fill: solid ? '#fff' : 'var(--text)', fontFamily: mono ? 'var(--font-mono)' : undefined }}
      >
        {label}
      </text>
    </g>
  );
}

/** Grid of cells (matrix / image / buffer). values in [0, 1] tint each cell. */
export function Cells({
  x,
  y,
  rows,
  cols,
  cell = 12,
  gap = 2,
  values,
  color,
  t = 1,
  highlight,
  highlightColor
}: {
  x: number;
  y: number;
  rows: number;
  cols: number;
  cell?: number;
  gap?: number;
  values?: (r: number, c: number) => number;
  color: string;
  t?: number;
  /** cells to outline, as "r,c" keys */
  highlight?: Set<string>;
  highlightColor?: string;
}) {
  const out: ReactNode[] = [];
  const shown = Math.ceil(rows * cols * Math.min(1, t * 1.15));
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < cols; k++) {
      const idx = r * cols + k;
      if (idx >= shown) continue;
      const v = values ? clamp(values(r, k), 0, 1) : 0.35;
      const on = highlight?.has(`${r},${k}`);
      out.push(
        <rect
          key={idx}
          x={x + k * (cell + gap)}
          y={y + r * (cell + gap)}
          width={cell}
          height={cell}
          rx={Math.min(3, cell / 4)}
          fill={alpha(color, 0.1 + 0.85 * v)}
          stroke={on ? highlightColor ?? color : 'none'}
          strokeWidth={on ? 2 : 0}
        />
      );
    }
  return <g>{out}</g>;
}

/** A dot travelling along an SVG path forever (skipped for reduced motion). */
export function Pulse({ path, color, dur = 1.6, r = 4.5, delay = 0 }: { path: string; color: string; dur?: number; r?: number; delay?: number }) {
  if (reducedMotion()) return null;
  return (
    <circle r={r} fill={color}>
      <animateMotion dur={`${dur}s`} begin={`${delay}s`} repeatCount="indefinite" path={path} />
    </circle>
  );
}

/** Speech-bubble-like rounded card with wrapped lines of text. */
export function Card({
  x,
  y,
  w,
  lines,
  color,
  t = 1,
  size = FS.s,
  title,
  mono = false
}: {
  x: number;
  y: number;
  w: number;
  lines: string[];
  color: string;
  t?: number;
  size?: number;
  title?: string;
  mono?: boolean;
}) {
  const lh = size * 1.45;
  const h = (title ? lh + 4 : 0) + lines.length * lh + 14;
  return (
    <g opacity={Math.min(1, t * 1.7)} transform={`translate(0 ${(1 - Math.min(1, t * 1.4)) * 8})`}>
      <rect x={x} y={y} width={w} height={h} rx={12} fill="var(--surface-solid)" stroke={color} strokeWidth={1.8} />
      {title && (
        <text x={x + 12} y={y + lh} style={{ fontSize: size, fontWeight: 850, fill: color }}>
          {title}
        </text>
      )}
      {lines.map((ln, i) => (
        <text
          key={i}
          x={x + 12}
          y={y + (title ? lh + 4 : 0) + lh * (i + 1)}
          style={{ fontSize: size, fontWeight: 600, fill: 'var(--text)', fontFamily: mono ? 'var(--font-mono)' : undefined }}
        >
          {ln}
        </text>
      ))}
    </g>
  );
}
