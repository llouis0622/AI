/*
 * Real (small) audio DSP for the lab 19 walkthrough (Worker F): a synthetic
 * "아 · 이 · 스" utterance, STFT, mel filterbank and log — computed in the browser.
 */

export const SR = 8000; // display sample rate (the lab uses 16 kHz; halved to keep it light)
export const DUR = 1.0;
export const WIN = 200; // 25 ms
export const HOP = 80; // 10 ms
export const NFFT = 256;
export const N_MELS = 40;

function rnd(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const bump = (f: number, center: number, width: number) => Math.exp(-0.5 * ((f - center) / width) ** 2);

/** Vowel-ish harmonic tone with two formants, plus a fricative noise burst. */
export const SIGNAL: Float64Array = (() => {
  const n = Math.round(SR * DUR);
  const x = new Float64Array(n);
  const r = rnd(9);
  const vowel = (t0: number, t1: number, f0a: number, f0b: number, F1: number, F2: number) => {
    let phase = 0;
    for (let i = Math.round(t0 * SR); i < Math.round(t1 * SR); i++) {
      const u = (i / SR - t0) / (t1 - t0);
      const f0 = f0a + (f0b - f0a) * u;
      phase += (2 * Math.PI * f0) / SR;
      const env = Math.sin(Math.PI * u) ** 0.6;
      let s = 0;
      for (let h = 1; h * f0 < SR / 2; h++) {
        const f = h * f0;
        const amp = 0.08 + bump(f, F1, 130) + 0.7 * bump(f, F2, 180);
        s += (amp / Math.sqrt(h)) * Math.sin(h * phase);
      }
      x[i] += 0.18 * env * s;
    }
  };
  vowel(0.05, 0.3, 150, 185, 800, 1200); // 아
  vowel(0.38, 0.62, 200, 225, 300, 2300); // 이
  // 스: band noise 2.4–3.8 kHz
  const comps = Array.from({ length: 48 }, () => ({ f: 2400 + r() * 1400, p: r() * Math.PI * 2 }));
  for (let i = Math.round(0.7 * SR); i < Math.round(0.92 * SR); i++) {
    const u = (i / SR - 0.7) / 0.22;
    const env = Math.sin(Math.PI * u) ** 0.8;
    let s = 0;
    for (const c of comps) s += Math.sin((2 * Math.PI * c.f * i) / SR + c.p);
    x[i] += 0.035 * env * s;
  }
  for (let i = 0; i < n; i++) x[i] += (r() - 0.5) * 0.004;
  return x;
})();

/** In-place iterative radix-2 FFT on (re, im). */
function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k);
        const wi = Math.sin(ang * k);
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
      }
    }
  }
}

/** Power spectrogram: frames × (NFFT/2 + 1), Hann window. */
export const SPEC: number[][] = (() => {
  const frames: number[][] = [];
  const hann = Array.from({ length: WIN }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1)));
  for (let start = 0; start + WIN <= SIGNAL.length; start += HOP) {
    const re = new Float64Array(NFFT);
    const im = new Float64Array(NFFT);
    for (let i = 0; i < WIN; i++) re[i] = SIGNAL[start + i] * hann[i];
    fft(re, im);
    const row: number[] = [];
    for (let k = 0; k <= NFFT / 2; k++) row.push(re[k] * re[k] + im[k] * im[k]);
    frames.push(row);
  }
  return frames;
})();

const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1);

/** Triangular mel filters: N_MELS × (NFFT/2 + 1). */
export const MEL_FB: number[][] = (() => {
  const nBins = NFFT / 2 + 1;
  const mMax = hzToMel(SR / 2);
  const pts = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz((mMax * i) / (N_MELS + 1)));
  const binHz = (k: number) => (k * SR) / NFFT;
  return Array.from({ length: N_MELS }, (_, m) => {
    const [lo, mid, hi] = [pts[m], pts[m + 1], pts[m + 2]];
    return Array.from({ length: nBins }, (_, k) => {
      const f = binHz(k);
      if (f <= lo || f >= hi) return 0;
      return f <= mid ? (f - lo) / (mid - lo) : (hi - f) / (hi - mid);
    });
  });
})();

/** Mel power: frames × N_MELS */
export const MEL: number[][] = SPEC.map((row) => MEL_FB.map((w) => w.reduce((s, wk, k) => s + wk * row[k], 0)));

export const MEL_CENTERS_HZ: number[] = (() => {
  const mMax = hzToMel(SR / 2);
  return Array.from({ length: N_MELS }, (_, i) => melToHz((mMax * (i + 1)) / (N_MELS + 1)));
})();

/** Normalise a frames × bins matrix to [0, 1] (optionally after log). */
export function normalise(mat: number[][], log: boolean): number[][] {
  const tf = (v: number) => (log ? Math.log(v + 1e-9) : v);
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of mat)
    for (const v of row) {
      const x = tf(v);
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
  // clip the floor for log so silence is dark but not dominating
  if (log) lo = Math.max(lo, hi - 12);
  return mat.map((row) => row.map((v) => Math.min(1, Math.max(0, (tf(v) - lo) / (hi - lo || 1)))));
}

/** Render frames × bins (low frequency at the bottom) to a PNG data URL, cached per key. */
const cache = new Map<string, string>();
export function heatmapURL(key: string, mat: number[][], color: (v: number) => [number, number, number]): string {
  const hit = cache.get(key);
  if (hit) return hit;
  if (typeof document === 'undefined') return '';
  const w = mat.length;
  const h = mat[0].length;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++)
    for (let y = 0; y < h; y++) {
      const [r, g, b] = color(mat[x][h - 1 - y]);
      const o = (y * w + x) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
  const url = canvas.toDataURL('image/png');
  cache.set(key, url);
  return url;
}

/** #rgb or #rrggbb → [r, g, b] (built CSS shortens colors like #ffffff to #fff). */
export function hexToRgb(hex: string, fallback: [number, number, number] = [124, 108, 246]): [number, number, number] {
  const m = hex.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!m) return fallback;
  let s = m[1];
  if (s.length === 3) s = s.split('').map((ch) => ch + ch).join('');
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}
