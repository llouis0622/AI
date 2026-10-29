/**
 * A 2D "smiley" made of small Gaussian blobs (a Gaussian mixture) for chapter 20.
 * Because the data distribution is known exactly, the ideal denoiser E[x0 | x_t] — and hence
 * the score / noise prediction a diffusion model would learn — can be computed in closed form.
 */
import { ALPHA_BAR, T } from './_pictures';

export const S_DATA = 0.08;
/** 0 = 눈, 1 = 입, 2 = 얼굴 테두리 */
export type Group = 0 | 1 | 2;

export const COMPONENTS: { m: [number, number]; g: Group }[] = (() => {
  const out: { m: [number, number]; g: Group }[] = [
    { m: [-0.42, 0.42], g: 0 },
    { m: [0.42, 0.42], g: 0 }
  ];
  for (let k = 0; k < 7; k++) {
    const th = ((205 + (k * 130) / 6) * Math.PI) / 180;
    out.push({ m: [0.72 * Math.cos(th), 0.08 + 0.72 * Math.sin(th)], g: 1 });
  }
  for (let k = 0; k < 16; k++) {
    const th = (k / 16) * 2 * Math.PI;
    out.push({ m: [1.22 * Math.cos(th), 1.22 * Math.sin(th)], g: 2 });
  }
  return out;
})();

const K = COMPONENTS.length;

/** Exact E[x0 | x_t] for the mixture, plus how sure we are which blob the point belongs to. */
export function denoise(x: number, y: number, t: number) {
  const ab = ALPHA_BAR[Math.max(1, t)];
  const sa = Math.sqrt(ab);
  const v = ab * S_DATA * S_DATA + 1 - ab;
  const gain = (sa * S_DATA * S_DATA) / v;
  let best = -Infinity;
  const lw = COMPONENTS.map(({ m }) => {
    const l = -((x - sa * m[0]) ** 2 + (y - sa * m[1]) ** 2) / (2 * v);
    if (l > best) best = l;
    return l;
  });
  let ws = 0;
  let mx = 0;
  let my = 0;
  let top = 0;
  let topK = 0;
  for (let k = 0; k < K; k++) {
    const w = Math.exp(lw[k] - best);
    const { m } = COMPONENTS[k];
    ws += w;
    mx += w * (m[0] + gain * (x - sa * m[0]));
    my += w * (m[1] + gain * (y - sa * m[1]));
    if (w > top) {
      top = w;
      topK = k;
    }
  }
  const conf = Math.max(0, (top / ws - 1 / K) / (1 - 1 / K));
  return { mx: mx / ws, my: my / ws, conf, group: COMPONENTS[topK].g };
}

/** Direction a point should move to become more "data-like" (∝ the score ∇log p_t). */
export function scoreDir(x: number, y: number, t: number): [number, number] {
  const ab = ALPHA_BAR[Math.max(1, t)];
  const { mx, my } = denoise(x, y, t);
  const sa = Math.sqrt(ab);
  return [(sa * mx - x) / (1 - ab), (sa * my - y) / (1 - ab)];
}

export function uniformSteps(n: number) {
  const ts: number[] = [];
  for (let i = n; i >= 1; i--) ts.push(Math.round((T * i) / n));
  return ts;
}

/** Ancestral DDPM step for one coordinate, given the predicted x0. */
export function ddpm(x: number, x0: number, t: number, tPrev: number, z: number) {
  const ab = ALPHA_BAR[t];
  const abPrev = ALPHA_BAR[tPrev];
  const beta = 1 - ab / abPrev;
  const c0 = (Math.sqrt(abPrev) * beta) / (1 - ab);
  const ct = (Math.sqrt(1 - beta) * (1 - abPrev)) / (1 - ab);
  return c0 * x0 + ct * x + Math.sqrt(((1 - abPrev) / (1 - ab)) * beta) * z;
}

/** Fraction of points sitting on the smiley (within 3σ of some blob). */
export function onShape(xs: Float32Array) {
  let ok = 0;
  const n = xs.length / 2;
  for (let i = 0; i < n; i++) {
    const x = xs[2 * i];
    const y = xs[2 * i + 1];
    for (const { m } of COMPONENTS)
      if ((x - m[0]) ** 2 + (y - m[1]) ** 2 < (3 * S_DATA) ** 2) {
        ok++;
        break;
      }
  }
  return ok / n;
}
