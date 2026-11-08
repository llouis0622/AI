import { useEffect, useState } from 'react';

export interface VizColors {
  theme: 'light' | 'dark';
  bg: string;
  panel: string;
  grid: string;
  axis: string;
  text: string;
  ink: string;
  /** Series colors 1..6 (index 0..5). */
  series: string[];
  pos: string;
  neg: string;
  low: string;
  mid: string;
  high: string;
  brand: string;
  brand2: string;
  border: string;
  surface: string;
}

// Server render and first paint use the default (dark) theme.
const FALLBACK: VizColors = {
  theme: 'dark',
  bg: '#1b1b1b',
  panel: '#222222',
  grid: 'rgba(255,255,255,.06)',
  axis: 'rgba(255,255,255,.36)',
  text: '#b5b3ae',
  ink: '#ecebe8',
  series: ['#86a8f5', '#e8a85a', '#6cc2b4', '#e48c9f', '#b6a0ee', '#bdb58a'],
  pos: '#86a8f5',
  neg: '#e8a85a',
  low: '#24324c',
  mid: '#4f6eae',
  high: '#a3bcf8',
  brand: '#86a8f5',
  brand2: '#e8a85a',
  border: '#2c2c2c',
  surface: '#1f1f1f'
};

export function readVizColors(): VizColors {
  if (typeof window === 'undefined') return FALLBACK;
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
  return {
    theme: document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark',
    bg: v('--viz-bg', FALLBACK.bg),
    panel: v('--viz-panel', FALLBACK.panel),
    grid: v('--viz-grid', FALLBACK.grid),
    axis: v('--viz-axis', FALLBACK.axis),
    text: v('--viz-text', FALLBACK.text),
    ink: v('--viz-ink', FALLBACK.ink),
    series: [1, 2, 3, 4, 5, 6].map((i) => v(`--viz-${i}`, FALLBACK.series[i - 1])),
    pos: v('--viz-pos', FALLBACK.pos),
    neg: v('--viz-neg', FALLBACK.neg),
    low: v('--viz-low', FALLBACK.low),
    mid: v('--viz-mid', FALLBACK.mid),
    high: v('--viz-high', FALLBACK.high),
    brand: v('--brand', FALLBACK.brand),
    brand2: v('--brand-2', FALLBACK.brand2),
    border: v('--border', FALLBACK.border),
    surface: v('--surface-solid', FALLBACK.surface)
  };
}

/** Theme-aware palette for canvas / SVG / WebGL; updates when the theme flips. */
export function useVizColors(): VizColors {
  const [colors, setColors] = useState<VizColors>(FALLBACK);
  useEffect(() => {
    setColors(readVizColors());
    const mo = new MutationObserver(() => setColors(readVizColors()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);
  return colors;
}

/** Mix two hex colors (t = 0 → a, 1 → b). Accepts #rgb/#rrggbb. */
export function mixHex(a: string, b: string, t: number): string {
  const pa = parseHex(a);
  const pb = parseHex(b);
  if (!pa || !pb) return t < 0.5 ? a : b;
  const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t));
  return `#${c.map((x) => x.toString(16).padStart(2, '0')).join('')}`;
}

function parseHex(h: string): [number, number, number] | null {
  const m = h.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!m) return null;
  let s = m[1];
  if (s.length === 3) s = s.split('').map((c) => c + c).join('');
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

/** Three-stop gradient low → mid → high for t in [0, 1]. */
export function ramp(c: Pick<VizColors, 'low' | 'mid' | 'high'>, t: number): string {
  const x = Math.min(1, Math.max(0, t));
  return x < 0.5 ? mixHex(c.low, c.mid, x * 2) : mixHex(c.mid, c.high, (x - 0.5) * 2);
}

/** Add alpha to a hex color: withAlpha('#ff0000', .3) → rgba(...) */
export function withAlpha(hex: string, alpha: number): string {
  const p = parseHex(hex);
  if (!p) return hex;
  return `rgba(${p[0]}, ${p[1]}, ${p[2]}, ${alpha})`;
}
