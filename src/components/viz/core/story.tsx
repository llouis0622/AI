import { useEffect, useState, type ReactNode } from 'react';
import { Button } from './ui';
import { useElementSize, useRafLoop, usePrefersReducedMotion } from './hooks';
import { useVizColors, withAlpha, type VizColors } from './colors';

/**
 * CodeStory — a step-by-step animated explainer for code labs.
 * Each step draws an SVG scene (in a fixed viewBox) next to the code it explains.
 *
 *   <CodeStory steps={[{ title, text, code, render: (t, c) => <g>…</g> }]} />
 *
 * `render(t, colors)` receives t ∈ [0, 1], the progress of the step's intro animation.
 */
export interface StoryStep {
  title: string;
  text: ReactNode;
  /** Short code excerpt (5–12 lines) that this step corresponds to. */
  code?: string;
  /** 1-based line numbers inside `code` to highlight. */
  highlight?: number[];
  render: (t: number, c: VizColors) => ReactNode;
}

export function CodeStory({
  steps,
  viewBox = [640, 300],
  stepSeconds = 1.4
}: {
  steps: StoryStep[];
  viewBox?: [number, number];
  /** Duration of each step's intro animation. */
  stepSeconds?: number;
}) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [i, setI] = useState(0);
  const [t, setT] = useState(1);
  const [auto, setAuto] = useState(false);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const wide = width > 720;

  useEffect(() => setT(reduce ? 1 : 0), [i, reduce]);
  useRafLoop((dt) => {
    setT((v) => {
      const nv = Math.min(1, v + dt / stepSeconds);
      return nv;
    });
  }, t < 1);

  useEffect(() => {
    if (!auto || t < 1) return;
    const id = window.setTimeout(() => {
      if (i < steps.length - 1) setI(i + 1);
      else setAuto(false);
    }, 1600);
    return () => window.clearTimeout(id);
  }, [auto, t, i, steps.length]);

  const step = steps[i];
  return (
    <div ref={box}>
      <div style={{ display: 'grid', gridTemplateColumns: wide && step.code ? 'minmax(0, 1.35fr) minmax(0, 1fr)' : '1fr' }}>
        <div style={{ padding: '14px 14px 6px' }}>
          <svg className="vz-svg" viewBox={`0 0 ${viewBox[0]} ${viewBox[1]}`} role="img" aria-label={step.title} style={{ touchAction: 'pan-y' }}>
            {step.render(easeOut(t), c)}
          </svg>
        </div>
        {step.code && (
          <div style={{ padding: wide ? '14px 14px 6px 0' : '0 14px 6px', minWidth: 0 }}>
            <CodeBlock code={step.code} highlight={step.highlight} c={c} />
          </div>
        )}
      </div>
      <div className="vz-pad" style={{ paddingTop: 6, display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', flexWrap: 'wrap' }}>
          <span
            style={{
              display: 'inline-grid',
              placeItems: 'center',
              minWidth: 30,
              height: 26,
              padding: '0 8px',
              borderRadius: 9,
              background: 'var(--grad-btn)',
              color: 'var(--btn-ink)',
              fontWeight: 850,
              fontSize: 13
            }}
          >
            {i + 1}/{steps.length}
          </span>
          <strong style={{ fontSize: 16.5, letterSpacing: '-0.02em' }}>{step.title}</strong>
        </div>
        <div style={{ fontSize: 14.5, lineHeight: 1.7, color: 'var(--text-2)' }}>{step.text}</div>
      </div>
      <div className="vz-controls" style={{ alignItems: 'center' }}>
        <Button onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>
          ← 이전
        </Button>
        <div style={{ display: 'flex', gap: 6, flex: 1, justifyContent: 'center', flexWrap: 'wrap' }} role="tablist" aria-label="단계">
          {steps.map((s, k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={k === i}
              aria-label={`${k + 1}단계: ${s.title}`}
              onClick={() => {
                setAuto(false);
                setI(k);
              }}
              style={{
                width: k === i ? 26 : 10,
                height: 10,
                borderRadius: 99,
                background: k === i ? c.brand : k < i ? withAlpha(hexOr(c.brand), 0.45) : 'var(--surface-3)',
                transition: 'width .3s, background-color .3s'
              }}
            />
          ))}
        </div>
        <Button variant="ghost" onClick={() => setAuto((a) => !a)}>
          {auto ? '❚❚ 멈춤' : '▶ 자동 재생'}
        </Button>
        <Button variant="primary" onClick={() => setI(Math.min(steps.length - 1, i + 1))} disabled={i === steps.length - 1}>
          다음 →
        </Button>
      </div>
    </div>
  );
}

const easeOut = (t: number) => 1 - (1 - t) ** 3;
const hexOr = (s: string) => (s.startsWith('#') ? s : '#7c6cf6');

const KEYWORDS = /\b(def|return|for|in|if|else|elif|import|from|as|class|with|while|lambda|yield|True|False|None|and|or|not|self|async|await)\b/g;

/** Tiny Python-ish highlighter: comments, strings, keywords, numbers. */
export function CodeBlock({ code, highlight = [], c }: { code: string; highlight?: number[]; c: VizColors }) {
  const lines = code.replace(/\n$/, '').split('\n');
  const dark = c.theme === 'dark';
  const colors = {
    kw: dark ? '#ff7ad9' : '#c2185b',
    str: dark ? '#b6ff3b' : '#2e7d32',
    num: dark ? '#ffb020' : '#b45309',
    com: dark ? '#6f7aa6' : '#9a94ad'
  };
  const paint = (line: string) => {
    const out: ReactNode[] = [];
    const commentAt = line.indexOf('#');
    const body = commentAt >= 0 ? line.slice(0, commentAt) : line;
    const comment = commentAt >= 0 ? line.slice(commentAt) : '';
    const parts = body.split(/("[^"]*"|'[^']*')/);
    parts.forEach((p, k) => {
      if (k % 2 === 1) out.push(<span key={`s${k}`} style={{ color: colors.str }}>{p}</span>);
      else {
        const seg = p.split(KEYWORDS);
        seg.forEach((q, j) => {
          if (j % 2 === 1) out.push(<span key={`k${k}-${j}`} style={{ color: colors.kw, fontWeight: 700 }}>{q}</span>);
          else {
            const nums = q.split(/(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)/);
            nums.forEach((n, m) =>
              out.push(
                m % 2 === 1 ? (
                  <span key={`n${k}-${j}-${m}`} style={{ color: colors.num }}>{n}</span>
                ) : (
                  <span key={`t${k}-${j}-${m}`}>{n}</span>
                )
              )
            );
          }
        });
      }
    });
    if (comment) out.push(<span key="c" style={{ color: colors.com, fontStyle: 'italic' }}>{comment}</span>);
    return out;
  };
  return (
    <pre
      style={{
        margin: 0,
        padding: '12px 0',
        borderRadius: 14,
        background: 'var(--code-bg)',
        border: '1px solid var(--border)',
        fontFamily: 'var(--font-mono)',
        fontSize: 12.5,
        lineHeight: 1.7,
        overflowX: 'auto',
        color: 'var(--code-text)'
      }}
    >
      {lines.map((ln, k) => {
        const on = highlight.includes(k + 1);
        return (
          <div
            key={k}
            style={{
              padding: '0 14px',
              whiteSpace: 'pre',
              background: on ? withAlpha(hexOr(c.brand), dark ? 0.18 : 0.12) : 'transparent',
              borderLeft: `3px solid ${on ? c.brand : 'transparent'}`
            }}
          >
            {ln ? paint(ln) : '\u00a0'}
          </div>
        );
      })}
    </pre>
  );
}

/* ---------- SVG building blocks for stories ---------- */

/** A labelled rounded box, e.g. a tensor or a module. */
export function Block({
  x,
  y,
  w,
  h,
  label,
  sub,
  color,
  t = 1,
  dashed = false
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: ReactNode;
  sub?: ReactNode;
  color: string;
  t?: number;
  dashed?: boolean;
}) {
  const s = 0.85 + 0.15 * t;
  return (
    <g opacity={Math.min(1, t * 1.6)} transform={`translate(${x + w / 2} ${y + h / 2}) scale(${s}) translate(${-w / 2} ${-h / 2})`}>
      <rect width={w} height={h} rx={12} fill={withAlpha(hexOr(color), 0.16)} stroke={color} strokeWidth={2} strokeDasharray={dashed ? '6 5' : undefined} />
      <text x={w / 2} y={sub ? h / 2 - 3 : h / 2 + 5} textAnchor="middle" style={{ fontSize: 14, fontWeight: 800, fill: 'var(--text)' }}>
        {label}
      </text>
      {sub && (
        <text x={w / 2} y={h / 2 + 15} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 650, fill: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
          {sub}
        </text>
      )}
    </g>
  );
}

/** Arrow that draws itself from (x1, y1) to (x2, y2) as t goes 0 → 1. */
export function FlowArrow({ x1, y1, x2, y2, color, t = 1, label }: { x1: number; y1: number; x2: number; y2: number; color: string; t?: number; label?: ReactNode }) {
  const xe = x1 + (x2 - x1) * t;
  const ye = y1 + (y2 - y1) * t;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const hl = 9;
  return (
    <g>
      <line x1={x1} y1={y1} x2={xe} y2={ye} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
      {t > 0.05 && (
        <path
          d={`M${xe} ${ye} L${xe - hl * Math.cos(ang - 0.45)} ${ye - hl * Math.sin(ang - 0.45)} L${xe - hl * Math.cos(ang + 0.45)} ${ye - hl * Math.sin(ang + 0.45)} Z`}
          fill={color}
        />
      )}
      {label && t > 0.6 && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 8} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 700, fill: 'var(--text-2)' }}>
          {label}
        </text>
      )}
    </g>
  );
}

/** A grid of small cells (a matrix / image / batch). values in [0, 1] tint the cells. */
export function Grid({
  x,
  y,
  rows,
  cols,
  cell = 14,
  values,
  color,
  t = 1,
  gap = 2
}: {
  x: number;
  y: number;
  rows: number;
  cols: number;
  cell?: number;
  values?: number[][];
  color: string;
  t?: number;
  gap?: number;
}) {
  const cells: ReactNode[] = [];
  const shown = Math.ceil(rows * cols * Math.min(1, t * 1.2));
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < cols; k++) {
      const idx = r * cols + k;
      if (idx >= shown) continue;
      const v = values?.[r]?.[k] ?? 0.35;
      cells.push(<rect key={idx} x={x + k * (cell + gap)} y={y + r * (cell + gap)} width={cell} height={cell} rx={3} fill={withAlpha(hexOr(color), 0.12 + 0.8 * v)} />);
    }
  return <g>{cells}</g>;
}
