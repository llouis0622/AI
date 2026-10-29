/* Tiny 10×10 digit bitmaps + helpers shared by labs 14 and 15 (Worker F). */

const RAW: Record<string, string[]> = {
  '3': ['..######..', '.##....##.', '.......##.', '.......##.', '...#####..', '.......##.', '.......##.', '.##....##.', '..######..', '..........'],
  '8': ['..######..', '.##....##.', '.##....##.', '..######..', '.##....##.', '.##....##.', '.##....##.', '..######..', '..........', '..........'],
  '1': ['....##....', '...###....', '..####....', '....##....', '....##....', '....##....', '....##....', '..######..', '..........', '..........'],
  '7': ['.########.', '.......##.', '......##..', '.....##...', '....##....', '...##.....', '...##.....', '...##.....', '..........', '..........'],
  '2': ['..######..', '.##....##.', '.......##.', '......##..', '....##....', '..##......', '.##.......', '.########.', '..........', '..........']
};

export type Img = number[][];

export function digit(d: keyof typeof RAW | string): Img {
  const rows = RAW[d] ?? RAW['3'];
  return rows.map((r) => [...r].map((ch) => (ch === '#' ? 1 : 0)));
}

/** 3×3 box blur, `times` passes. */
export function blur(img: Img, times = 1): Img {
  let cur = img;
  for (let k = 0; k < times; k++) {
    const n = cur.length;
    const m = cur[0].length;
    const out: Img = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < m; j++) {
        let s = 0;
        let c = 0;
        for (let di = -1; di <= 1; di++)
          for (let dj = -1; dj <= 1; dj++) {
            const ii = i + di;
            const jj = j + dj;
            if (ii >= 0 && ii < n && jj >= 0 && jj < m) {
              s += cur[ii][jj] * (di === 0 && dj === 0 ? 2 : 1);
              c += di === 0 && dj === 0 ? 2 : 1;
            }
          }
        row.push(s / c);
      }
      out.push(row);
    }
    cur = out;
  }
  return cur;
}

/** Deterministic standard-normal noise image. */
export function noise(seed: number, n = 10, m = 10): Img {
  let a = seed >>> 0;
  const rnd = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const g = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
  return Array.from({ length: n }, () => Array.from({ length: m }, g));
}

/** x_t = √ᾱ·x0 + √(1-ᾱ)·ε with x0 mapped to [-1, 1]; returns values back in [0, 1] for drawing. */
export function diffuse(x0: Img, eps: Img, alphaBar: number): Img {
  const a = Math.sqrt(alphaBar);
  const b = Math.sqrt(1 - alphaBar);
  return x0.map((row, i) => row.map((v, j) => clamp01((a * (v * 2 - 1) + b * eps[i][j] + 1) / 2)));
}

export function mix(a: Img, b: Img, t: number): Img {
  return a.map((row, i) => row.map((v, j) => v * (1 - t) + b[i][j] * t));
}

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** DDPM linear schedule (T = 1000): β from 1e-4 to 0.02 and ᾱ = cumulative product of (1 - β). */
export const T_STEPS = 1000;
export const BETA = Array.from({ length: T_STEPS }, (_, i) => 1e-4 + ((0.02 - 1e-4) * i) / (T_STEPS - 1));
export const ALPHA_BAR = (() => {
  const out: number[] = [];
  let p = 1;
  for (const b of BETA) {
    p *= 1 - b;
    out.push(p);
  }
  return out;
})();
