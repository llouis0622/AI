/**
 * A real (tiny) 1D GAN for chapter 19.
 * Discriminator: logistic regression on Gaussian bump features.
 * Generator: a mixture of K Gaussians, trained through its samples
 * (reparameterization), with the non-saturating generator loss.
 */
import { rng } from '../core/math';

export interface GanConfig {
  K: number;
  init: [number, number][];
  /** width of the discriminator's bump features (smaller = sharper, stronger) */
  width: number;
  lrD: number;
  /** discriminator updates per generator update */
  kD: number;
  /** weight decay on the discriminator (smoothness) */
  lam: number;
  lrG: number;
}

export const REAL = [
  { m: -2, s: 0.55, w: 0.5 },
  { m: 2.2, s: 0.55, w: 0.5 }
];

const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
export const pReal = (x: number) => REAL.reduce((a, c) => a + c.w * npdf(x, c.m, c.s), 0);

export class TinyGan {
  readonly cfg: GanConfig;
  readonly centers: number[];
  w: number[];
  b = 0;
  gen: { m: number; ls: number }[];
  steps = 0;
  lastReal: number[] = [];
  lastFake: number[] = [];
  private R: ReturnType<typeof rng>;

  constructor(cfg: GanConfig, seed = 3) {
    this.cfg = cfg;
    const J = Math.round(12 / cfg.width) + 1;
    this.centers = Array.from({ length: J }, (_, j) => -6 + (12 * j) / (J - 1));
    this.w = new Array(J).fill(0);
    this.gen = cfg.init.map(([m, s]) => ({ m, ls: Math.log(s) }));
    this.R = rng(seed);
    this.sampleBatches();
  }

  private feats(x: number) {
    const { width } = this.cfg;
    return this.centers.map((c) => Math.exp(-((x - c) ** 2) / (2 * width * width)));
  }

  score(x: number) {
    const f = this.feats(x);
    let s = this.b;
    for (let j = 0; j < f.length; j++) s += this.w[j] * f[j];
    return { s, f };
  }

  D(x: number) {
    return 1 / (1 + Math.exp(-this.score(x).s));
  }

  pGen(x: number) {
    return this.gen.reduce((a, c) => a + npdf(x, c.m, Math.exp(c.ls)) / this.gen.length, 0);
  }

  private sampleReal() {
    const c = this.R.next() < 0.5 ? REAL[0] : REAL[1];
    return c.m + c.s * this.R.normal();
  }

  private sampleFake() {
    const k = Math.floor(this.R.next() * this.gen.length);
    const e = this.R.normal();
    return { x: this.gen[k].m + Math.exp(this.gen[k].ls) * e, k, e };
  }

  private sampleBatches() {
    this.lastReal = Array.from({ length: 40 }, () => this.sampleReal());
    this.lastFake = Array.from({ length: 40 }, () => this.sampleFake().x);
  }

  step() {
    const { kD, lrD, lam, lrG, width } = this.cfg;
    const J = this.w.length;
    for (let k = 0; k < kD; k++) {
      const gw = new Array(J).fill(0);
      let gb = 0;
      const n = 96;
      for (let i = 0; i < n; i++) {
        const sr = this.score(this.sampleReal());
        const dr = 1 / (1 + Math.exp(-sr.s));
        for (let j = 0; j < J; j++) gw[j] += ((1 - dr) * sr.f[j]) / n;
        gb += (1 - dr) / n;
        const sf = this.score(this.sampleFake().x);
        const df = 1 / (1 + Math.exp(-sf.s));
        for (let j = 0; j < J; j++) gw[j] -= (df * sf.f[j]) / n;
        gb -= df / n;
      }
      for (let j = 0; j < J; j++) this.w[j] += lrD * (gw[j] - lam * this.w[j]);
      this.b += lrD * gb;
    }
    const gm = this.gen.map(() => 0);
    const gs = this.gen.map(() => 0);
    const cnt = this.gen.map(() => 0);
    for (let i = 0; i < 128; i++) {
      const { x, k, e } = this.sampleFake();
      const { s, f } = this.score(x);
      const d = 1 / (1 + Math.exp(-s));
      let ds = 0;
      for (let j = 0; j < J; j++) ds += this.w[j] * f[j] * (-(x - this.centers[j]) / (width * width));
      const grad = (1 - d) * ds;
      gm[k] += grad;
      gs[k] += grad * e * Math.exp(this.gen[k].ls);
      cnt[k]++;
    }
    this.gen.forEach((c, k) => {
      if (!cnt[k]) return;
      c.m += (lrG * gm[k]) / cnt[k];
      c.ls = Math.max(Math.log(0.12), Math.min(Math.log(3), c.ls + (lrG * gs[k]) / cnt[k]));
    });
    this.steps++;
    this.sampleBatches();
  }

  /** ∫ min(p_real, p_gen) dx — 1 means the fakes are indistinguishable. */
  overlap() {
    let o = 0;
    const dx = 0.03;
    for (let x = -7; x <= 9; x += dx) o += Math.min(pReal(x), this.pGen(x)) * dx;
    return o;
  }

  /** How many of the two real peaks get at least a fair share of fakes. */
  modesCovered() {
    const near = REAL.map((c) => {
      let mass = 0;
      const dx = 0.03;
      for (let x = c.m - 1.2; x <= c.m + 1.2; x += dx) mass += this.pGen(x) * dx;
      return mass;
    });
    return near.filter((m) => m > 0.25).length;
  }

  accuracy() {
    let acc = 0;
    this.lastReal.forEach((x) => (acc += this.D(x) >= 0.5 ? 1 : 0));
    this.lastFake.forEach((x) => (acc += this.D(x) < 0.5 ? 1 : 0));
    return acc / (this.lastReal.length + this.lastFake.length);
  }
}

export const SCENARIOS: Record<'good' | 'strong' | 'collapse', { label: string; cfg: GanConfig }> = {
  good: { label: '😊 잘 되는 경우', cfg: { K: 2, init: [[-0.6, 0.8], [0.9, 0.8]], width: 0.5, lrD: 0.4, kD: 2, lam: 0.01, lrG: 0.06 } },
  strong: { label: '💪 판별기가 너무 셀 때', cfg: { K: 2, init: [[4.4, 0.4], [5.0, 0.4]], width: 0.18, lrD: 2.0, kD: 10, lam: 0, lrG: 0.12 } },
  collapse: { label: '🔁 모드 붕괴', cfg: { K: 1, init: [[1.2, 0.6]], width: 0.5, lrD: 0.4, kD: 2, lam: 0.01, lrG: 0.05 } }
};
