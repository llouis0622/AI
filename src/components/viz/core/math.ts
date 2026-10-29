/** Small numeric helpers shared by visualizations. */

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const invLerp = (a: number, b: number, v: number) => (b === a ? 0 : (v - a) / (b - a));
export const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0);
export const mean = (xs: number[]) => (xs.length ? sum(xs) / xs.length : 0);
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

export function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

export function softmax(xs: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp((x - m) / t));
  const s = sum(e);
  return e.map((v) => v / s);
}

/** Seeded PRNG (mulberry32) so demos look the same on every visit. */
export function rng(seed = 42) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(next(), 1e-12);
    const v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { next, normal, int: (n: number) => Math.floor(next() * n), pick: <T,>(xs: T[]) => xs[Math.floor(next() * xs.length)] };
}

/** Pretty number for readouts: fixed decimals (∞ for overflow). */
export function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return v > 0 ? '∞' : v < 0 ? '-∞' : '—';
  // keep the ASCII minus: the web font subset has no U+2212 glyph
  return Math.abs(v) >= 1e5 ? v.toExponential(1) : v.toFixed(digits);
}

export function pct(v: number, digits = 0): string {
  return `${(v * 100).toFixed(digits)}%`;
}

/**
 * Marching squares: line segments of f(x, y) = level on a res×res grid.
 * Returns flat [x0, y0, x1, y1, ...] in function coordinates.
 */
export function contourSegments(
  f: (x: number, y: number) => number,
  xr: [number, number],
  yr: [number, number],
  level: number,
  res = 60
): number[] {
  const out: number[] = [];
  const dx = (xr[1] - xr[0]) / res;
  const dy = (yr[1] - yr[0]) / res;
  const grid: number[][] = [];
  for (let j = 0; j <= res; j++) {
    const row: number[] = [];
    for (let i = 0; i <= res; i++) row.push(f(xr[0] + i * dx, yr[0] + j * dy));
    grid.push(row);
  }
  const interp = (x0: number, y0: number, v0: number, x1: number, y1: number, v1: number): [number, number] => {
    const t = v1 === v0 ? 0.5 : (level - v0) / (v1 - v0);
    return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t];
  };
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const x0 = xr[0] + i * dx;
      const y0 = yr[0] + j * dy;
      const x1 = x0 + dx;
      const y1 = y0 + dy;
      const a = grid[j][i];
      const b = grid[j][i + 1];
      const c = grid[j + 1][i + 1];
      const d = grid[j + 1][i];
      const idx = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const pts: [number, number][] = [];
      const edges: [number, number, number, number, number, number][] = [
        [x0, y0, a, x1, y0, b],
        [x1, y0, b, x1, y1, c],
        [x1, y1, c, x0, y1, d],
        [x0, y1, d, x0, y0, a]
      ];
      for (const [ex0, ey0, ev0, ex1, ey1, ev1] of edges) {
        if ((ev0 > level) !== (ev1 > level)) pts.push(interp(ex0, ey0, ev0, ex1, ey1, ev1));
      }
      if (pts.length === 2) out.push(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
      else if (pts.length === 4) out.push(...pts[0], ...pts[1], ...pts[2], ...pts[3]);
    }
  }
  return out;
}
