/**
 * Tiny procedurally drawn pictures (32×32 RGB, values in [-1, 1]) and the DDPM math for chapter 20.
 * The "noise predictor" here is exact: with only a handful of training pictures, the best possible
 * guess of the original x0 given a noisy x_t is a weighted average of those pictures (the posterior mean).
 */

export const SIZE = 32;
export const T = 1000;

type RGB = [number, number, number];
type SDF = (x: number, y: number) => number;

const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

class Raster {
  px = new Float32Array(SIZE * SIZE * 3);
  fill(f: (x: number, y: number) => RGB) {
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const c = f(x + 0.5, y + 0.5);
        const i = (y * SIZE + x) * 3;
        this.px[i] = c[0];
        this.px[i + 1] = c[1];
        this.px[i + 2] = c[2];
      }
    return this;
  }
  /** Paint a shape given by a signed distance function, with 1px anti-aliasing. */
  shape(sdf: SDF, color: string, alpha = 1) {
    const c = hex(color);
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const a = clamp01(0.5 - sdf(x + 0.5, y + 0.5)) * alpha;
        if (a <= 0) continue;
        const i = (y * SIZE + x) * 3;
        for (let k = 0; k < 3; k++) this.px[i + k] = this.px[i + k] * (1 - a) + c[k] * a;
      }
    return this;
  }
  signed() {
    return this.px.map((v) => v * 2 - 1);
  }
}

const circle = (cx: number, cy: number, r: number): SDF => (x, y) => Math.hypot(x - cx, y - cy) - r;
const box = (x0: number, y0: number, x1: number, y1: number): SDF => {
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const hx = (x1 - x0) / 2;
  const hy = (y1 - y0) / 2;
  return (x, y) => {
    const qx = Math.abs(x - cx) - hx;
    const qy = Math.abs(y - cy) - hy;
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
  };
};
const segment = (ax: number, ay: number, bx: number, by: number, r: number): SDF => (x, y) => {
  const pax = x - ax;
  const pay = y - ay;
  const bax = bx - ax;
  const bay = by - ay;
  const h = clamp01((pax * bax + pay * bay) / (bax * bax + bay * bay));
  return Math.hypot(pax - bax * h, pay - bay * h) - r;
};
const triangle = (p: [number, number][]): SDF => (x, y) => {
  let d = Infinity;
  let inside = true;
  const sgn = Math.sign((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]));
  for (let k = 0; k < 3; k++) {
    const [ax, ay] = p[k];
    const [bx, by] = p[(k + 1) % 3];
    d = Math.min(d, segment(ax, ay, bx, by, 0)(x, y));
    if (sgn * ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) < 0) inside = false;
  }
  return inside ? -d : d;
};
/** Heart with its bottom tip at (cx, tipY) and height ≈ s. */
const heart = (cx: number, tipY: number, s: number): SDF => (x, y) => {
  const px = Math.abs(x - cx) / s;
  const py = (tipY - y) / s;
  let d: number;
  if (py + px > 1) d = Math.hypot(px - 0.25, py - 0.75) - Math.SQRT2 / 4;
  else {
    const m = 0.5 * Math.max(px + py, 0);
    d = Math.sqrt(Math.min((px - 0) ** 2 + (py - 1) ** 2, (px - m) ** 2 + (py - m) ** 2)) * Math.sign(px - py);
  }
  return d * s;
};
const vgrad = (top: string, bottom: string) => {
  const a = hex(top);
  const b = hex(bottom);
  return (_x: number, y: number): RGB => {
    const t = y / SIZE;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  };
};

function sun() {
  const r = new Raster().fill(vgrad('#6ec3ff', '#d8f1ff'));
  for (let k = 0; k < 8; k++) {
    const a = (k * Math.PI) / 4 + Math.PI / 8;
    r.shape(segment(16 + 9.6 * Math.cos(a), 16 + 9.6 * Math.sin(a), 16 + 13.2 * Math.cos(a), 16 + 13.2 * Math.sin(a), 1.15), '#ffae00');
  }
  r.shape(circle(16, 16, 7.6), '#ffc93c');
  r.shape(circle(13.4, 14.6, 1), '#7a4500').shape(circle(18.6, 14.6, 1), '#7a4500');
  r.shape((x, y) => Math.max(Math.abs(Math.hypot(x - 16, y - 15.2) - 3.7) - 0.6, 17.3 - y), '#7a4500');
  return r.signed();
}

function house() {
  const r = new Raster().fill(vgrad('#9fd8ff', '#e6f6ff'));
  r.shape(box(0, 25, 32, 32), '#5dbb63');
  r.shape(box(20, 8, 22.6, 14), '#b8473f');
  r.shape(box(9, 15.5, 23, 26), '#ffe2b3');
  r.shape(triangle([[6.2, 16.6], [25.8, 16.6], [16, 6.8]]), '#e5484d');
  r.shape(box(13.9, 19.4, 18.1, 26), '#8b5a3c');
  r.shape(box(19.4, 17.6, 22.2, 20.4), '#4fb0ff');
  r.shape(box(10, 17.6, 12.8, 20.4), '#4fb0ff');
  return r.signed();
}

function cat() {
  const r = new Raster().fill(vgrad('#eee5ff', '#ddd0ff'));
  const fur = '#f39a4a';
  r.shape(triangle([[6.8, 15.5], [8.6, 3.8], [15, 9.6]]), fur).shape(triangle([[25.2, 15.5], [23.4, 3.8], [17, 9.6]]), fur);
  r.shape(triangle([[8.8, 12.5], [9.6, 6.6], [13.2, 9.8]]), '#ffb3c6').shape(triangle([[23.2, 12.5], [22.4, 6.6], [18.8, 9.8]]), '#ffb3c6');
  r.shape(circle(16, 18.6, 10), fur);
  r.shape(circle(16, 22.4, 4.6), '#ffe3c8');
  r.shape(circle(12, 17.4, 1.8), '#2a2233').shape(circle(20, 17.4, 1.8), '#2a2233');
  r.shape(circle(12.5, 16.8, 0.55), '#ffffff').shape(circle(20.5, 16.8, 0.55), '#ffffff');
  r.shape(triangle([[14.6, 20.4], [17.4, 20.4], [16, 22]]), '#ff6f91');
  for (const [ax, ay, bx, by] of [
    [10.5, 21.2, 3.5, 20],
    [10.5, 22.6, 3.8, 23.6],
    [21.5, 21.2, 28.5, 20],
    [21.5, 22.6, 28.2, 23.6]
  ])
    r.shape(segment(ax, ay, bx, by, 0.32), '#6b3f1f');
  return r.signed();
}

function heartPic() {
  const r = new Raster().fill(vgrad('#fff3f8', '#ffd9e8'));
  r.shape(heart(16, 27.5, 21), '#ff3f7f');
  r.shape(circle(11.4, 12.4, 2.2), '#ffc2d8', 0.9);
  return r.signed();
}

export interface Picture {
  id: string;
  label: string;
  emoji: string;
  data: Float32Array;
}

let cache: Picture[] | null = null;
export function pictures(): Picture[] {
  if (!cache)
    cache = [
      { id: 'sun', label: '해', emoji: '🌞', data: sun() },
      { id: 'house', label: '집', emoji: '🏠', data: house() },
      { id: 'cat', label: '고양이', emoji: '🐱', data: cat() },
      { id: 'heart', label: '하트', emoji: '💗', data: heartPic() }
    ];
  return cache;
}

/** ᾱ_t for the classic DDPM linear schedule (β from 1e-4 to 0.02, T = 1000). ALPHA_BAR[0] = 1. */
export const ALPHA_BAR = (() => {
  const ab = new Float64Array(T + 1);
  ab[0] = 1;
  for (let t = 1; t <= T; t++) ab[t] = ab[t - 1] * (1 - (1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1)));
  return ab;
})();

/** x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε */
export function noisy(x0: Float32Array, eps: Float32Array, t: number, out = new Float32Array(x0.length)) {
  const a = Math.sqrt(ALPHA_BAR[t]);
  const b = Math.sqrt(1 - ALPHA_BAR[t]);
  for (let i = 0; i < x0.length; i++) out[i] = a * x0[i] + b * eps[i];
  return out;
}

/** Exact posterior over the training pictures given x_t, and the implied best guess of x0. */
export function guess(xt: Float32Array, t: number, pics: Float32Array[]) {
  const ab = ALPHA_BAR[Math.max(1, t)];
  const sa = Math.sqrt(ab);
  const v = 1 - ab;
  const logw = pics.map((p) => {
    let d = 0;
    for (let i = 0; i < p.length; i++) {
      const e = xt[i] - sa * p[i];
      d += e * e;
    }
    return -d / (2 * v);
  });
  const m = Math.max(...logw);
  const w = logw.map((l) => Math.exp(l - m));
  const s = w.reduce((a, b) => a + b, 0);
  const probs = w.map((x) => x / s);
  const x0 = new Float32Array(xt.length);
  pics.forEach((p, k) => {
    if (probs[k] < 1e-6) return;
    for (let i = 0; i < p.length; i++) x0[i] += probs[k] * p[i];
  });
  return { probs, x0 };
}

/** The reverse-process time steps: denser near t = T, where the picture is decided. */
export function reverseSteps(n: number) {
  const ts: number[] = [];
  for (let i = n; i >= 1; i--) ts.push(Math.max(1, Math.round(T * (i / n) ** 0.55)));
  return [...new Set(ts)];
}

/**
 * One ancestral DDPM step from t to tPrev using the predicted x0:
 * mean of q(x_{tPrev} | x_t, x0) plus fresh noise (pass z = null to skip the noise).
 */
export function reverseStep(xt: Float32Array, x0: Float32Array, t: number, tPrev: number, z: Float32Array | null) {
  const ab = ALPHA_BAR[t];
  const abPrev = ALPHA_BAR[tPrev];
  const beta = 1 - ab / abPrev;
  const c0 = (Math.sqrt(abPrev) * beta) / (1 - ab);
  const ct = (Math.sqrt(1 - beta) * (1 - abPrev)) / (1 - ab);
  const sigma = Math.sqrt(((1 - abPrev) / (1 - ab)) * beta);
  const out = new Float32Array(xt.length);
  for (let i = 0; i < xt.length; i++) out[i] = c0 * x0[i] + ct * xt[i] + (z ? sigma * z[i] : 0);
  return out;
}
