/*
 * A small but real DQN for CartPole-v1, written for the browser (lab 16, Worker F).
 * Physics follow gymnasium's CartPole-v1 exactly; the agent uses the lab's recipe:
 * ε-greedy, replay buffer, target network, Huber loss, Adam, grad-norm clipping.
 * Kept dependency-free so it can also run in Node for testing.
 *
 * Training is chaotic: a one-bit difference anywhere grows until two runs no longer
 * look alike. Engines may round Math.sin/cos/exp/log/pow differently, so this file
 * only uses + - * / and sqrt (exactly rounded everywhere) — the same seed gives the
 * same run in every browser.
 */

/* ---------------- deterministic math ---------------- */

const TWO_PI = 2 * Math.PI;
const reduceAngle = (x: number) => (x > Math.PI || x < -Math.PI ? x - TWO_PI * Math.round(x / TWO_PI) : x);

/** sin via its Taylor series up to x^21 (|x| ≤ π after reduction; CartPole keeps |θ| < 0.3). */
export function dsin(x: number): number {
  x = reduceAngle(x);
  const x2 = x * x;
  let s = 1;
  for (let k = 10; k >= 1; k--) s = 1 - (x2 / (2 * k * (2 * k + 1))) * s;
  return x * s;
}

/** cos via its Taylor series up to x^20. */
export function dcos(x: number): number {
  x = reduceAngle(x);
  const x2 = x * x;
  let s = 1;
  for (let k = 10; k >= 1; k--) s = 1 - (x2 / ((2 * k - 1) * 2 * k)) * s;
  return s;
}

const LN2 = 0.6931471805599453;
/** exp(x) = 2^k · e^r with |r| ≤ ln2/2. */
export function dexp(x: number): number {
  const k = Math.round(x / LN2);
  const r = x - k * LN2;
  let s = 1;
  for (let n = 16; n >= 1; n--) s = 1 + (r / n) * s;
  let p = 1;
  for (let i = 0; i < Math.abs(k); i++) p = k > 0 ? p * 2 : p / 2;
  return s * p;
}
const LN_099 = -0.01005033585350145; // ln(0.99)

/* ---------------- environment ---------------- */

const GRAVITY = 9.8;
const MASSCART = 1.0;
const MASSPOLE = 0.1;
const TOTAL_MASS = MASSCART + MASSPOLE;
const LENGTH = 0.5; // half the pole's length
const POLEMASS_LENGTH = MASSPOLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU = 0.02;
export const X_LIMIT = 2.4;
export const THETA_LIMIT = (12 * 2 * Math.PI) / 360;
export const MAX_STEPS = 500;

export type Rand = () => number;

export function mulberry(seed: number): Rand {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class CartPole {
  /** [x, x_dot, theta, theta_dot] */
  s: number[] = [0, 0, 0, 0];
  t = 0;
  rand: Rand;
  constructor(rand: Rand) {
    this.rand = rand;
    this.reset();
  }
  reset(): number[] {
    this.s = [0, 0, 0, 0].map(() => this.rand() * 0.1 - 0.05);
    this.t = 0;
    return this.s.slice();
  }
  step(action: number) {
    let [x, xDot, th, thDot] = this.s;
    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    const cos = dcos(th);
    const sin = dsin(th);
    const temp = (force + POLEMASS_LENGTH * thDot * thDot * sin) / TOTAL_MASS;
    const thAcc = (GRAVITY * sin - cos * temp) / (LENGTH * (4 / 3 - (MASSPOLE * cos * cos) / TOTAL_MASS));
    const xAcc = temp - (POLEMASS_LENGTH * thAcc * cos) / TOTAL_MASS;
    x += TAU * xDot;
    xDot += TAU * xAcc;
    th += TAU * thDot;
    thDot += TAU * thAcc;
    this.s = [x, xDot, th, thDot];
    this.t += 1;
    const terminated = x < -X_LIMIT || x > X_LIMIT || th < -THETA_LIMIT || th > THETA_LIMIT;
    const truncated = !terminated && this.t >= MAX_STEPS;
    return { s: this.s.slice(), r: 1, terminated, truncated };
  }
}

/* ---------------- tiny MLP: 4 → H → H → 2 (ReLU) ---------------- */

const SCALE = [2.4, 3, THETA_LIMIT, 3]; // input normalisation for a small net

export class QNet {
  h: number;
  // weights stored row-major: W1[h][4], W2[h][h], W3[2][h]
  p: Float64Array;
  sizes: { w1: number; b1: number; w2: number; b2: number; w3: number; b3: number };
  constructor(h: number, rand?: Rand) {
    this.h = h;
    const w1 = h * 4;
    const b1 = h;
    const w2 = h * h;
    const b2 = h;
    const w3 = 2 * h;
    const b3 = 2;
    this.sizes = { w1, b1, w2, b2, w3, b3 };
    this.p = new Float64Array(w1 + b1 + w2 + b2 + w3 + b3);
    if (rand) {
      // He-style init with a uniform draw of the same spread: U(-a, a) has std a/√3
      const uni = (std: number) => (rand() * 2 - 1) * std * Math.sqrt(3);
      let o = 0;
      for (let i = 0; i < w1; i++) this.p[o + i] = uni(Math.sqrt(2 / 4));
      o += w1 + b1;
      for (let i = 0; i < w2; i++) this.p[o + i] = uni(Math.sqrt(2 / h));
      o += w2 + b2;
      for (let i = 0; i < w3; i++) this.p[o + i] = uni(Math.sqrt(1 / h));
    }
  }
  copyFrom(other: QNet) {
    this.p.set(other.p);
  }
  /** Forward one state; returns [Q(s,0), Q(s,1)]. Optionally records activations. */
  forward(s: number[], acts?: { x: number[]; a1: number[]; a2: number[] }): [number, number] {
    const h = this.h;
    const p = this.p;
    const { w1, b1, w2, b2, w3 } = this.sizes;
    const x = [s[0] / SCALE[0], s[1] / SCALE[1], s[2] / SCALE[2], s[3] / SCALE[3]];
    const a1 = new Array<number>(h);
    const o1 = w1;
    for (let j = 0; j < h; j++) {
      let z = p[o1 + j];
      const r = j * 4;
      z += p[r] * x[0] + p[r + 1] * x[1] + p[r + 2] * x[2] + p[r + 3] * x[3];
      a1[j] = z > 0 ? z : 0;
    }
    const oW2 = w1 + b1;
    const oB2 = oW2 + w2;
    const a2 = new Array<number>(h);
    for (let j = 0; j < h; j++) {
      let z = p[oB2 + j];
      const r = oW2 + j * h;
      for (let k = 0; k < h; k++) z += p[r + k] * a1[k];
      a2[j] = z > 0 ? z : 0;
    }
    const oW3 = oB2 + b2;
    const oB3 = oW3 + w3;
    let q0 = p[oB3];
    let q1 = p[oB3 + 1];
    for (let k = 0; k < h; k++) {
      q0 += p[oW3 + k] * a2[k];
      q1 += p[oW3 + h + k] * a2[k];
    }
    if (acts) {
      acts.x = x;
      acts.a1 = a1;
      acts.a2 = a2;
    }
    return [q0, q1];
  }
  /** Accumulate dLoss/dParams for one sample where only output `a` has gradient `g`. */
  backward(acts: { x: number[]; a1: number[]; a2: number[] }, a: number, g: number, grad: Float64Array) {
    const h = this.h;
    const p = this.p;
    const { w1, b1, w2, b2, w3 } = this.sizes;
    const oW2 = w1 + b1;
    const oB2 = oW2 + w2;
    const oW3 = oB2 + b2;
    const oB3 = oW3 + w3;
    // output layer
    const rowW3 = oW3 + a * h;
    grad[oB3 + a] += g;
    const d2 = new Array<number>(h);
    for (let k = 0; k < h; k++) {
      grad[rowW3 + k] += g * acts.a2[k];
      d2[k] = acts.a2[k] > 0 ? g * p[rowW3 + k] : 0;
    }
    // hidden layer 2
    const d1 = new Array<number>(h).fill(0);
    for (let j = 0; j < h; j++) {
      const dj = d2[j];
      if (dj === 0) continue;
      grad[oB2 + j] += dj;
      const r = oW2 + j * h;
      for (let k = 0; k < h; k++) {
        grad[r + k] += dj * acts.a1[k];
        d1[k] += dj * p[r + k];
      }
    }
    // hidden layer 1
    for (let k = 0; k < h; k++) {
      const dk = acts.a1[k] > 0 ? d1[k] : 0;
      if (dk === 0) continue;
      grad[w1 + k] += dk;
      const r = k * 4;
      grad[r] += dk * acts.x[0];
      grad[r + 1] += dk * acts.x[1];
      grad[r + 2] += dk * acts.x[2];
      grad[r + 3] += dk * acts.x[3];
    }
  }
}

/* ---------------- agent ---------------- */

export interface Transition {
  s: number[];
  a: number;
  r: number;
  s2: number[];
  done: number;
}

export interface DqnConfig {
  hidden: number;
  gamma: number;
  lr: number;
  batch: number;
  bufferSize: number;
  learnStart: number;
  epsStart: number;
  epsEnd: number;
  epsDecay: number;
  targetSync: number;
  seed: number;
  /** Double DQN: pick a' with the online net, evaluate it with the target net (lab extension 1). */
  double: boolean;
}

export const DEFAULT_CONFIG: DqnConfig = {
  hidden: 32,
  gamma: 0.99,
  lr: 1e-3,
  batch: 32,
  bufferSize: 20000,
  learnStart: 500,
  epsStart: 1,
  epsEnd: 0.05,
  epsDecay: 1500,
  targetSync: 500,
  seed: 7,
  double: false
};

export class DqnAgent {
  cfg: DqnConfig;
  rand: Rand;
  env: CartPole;
  q: QNet;
  target: QNet;
  buffer: Transition[] = [];
  bufIdx = 0;
  step = 0;
  episode = 0;
  epReward = 0;
  episodeRewards: number[] = [];
  syncs = 0;
  lastLoss = 0;
  /** indices of the last sampled minibatch (for visualisation) */
  lastBatch: number[] = [];
  lastAction = 0;
  lastExplore = false;
  /** the state the last step landed in and how it ended (for drawing the fall) */
  last: { s2: number[]; terminated: boolean; truncated: boolean } | null = null;
  s: number[];
  private m: Float64Array;
  private v: Float64Array;
  private grad: Float64Array;
  private b1t = 1;
  private b2t = 1;

  constructor(cfg: Partial<DqnConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
    this.rand = mulberry(this.cfg.seed);
    this.env = new CartPole(mulberry(this.cfg.seed + 1000));
    this.q = new QNet(this.cfg.hidden, this.rand);
    this.target = new QNet(this.cfg.hidden);
    this.target.copyFrom(this.q);
    const n = this.q.p.length;
    this.m = new Float64Array(n);
    this.v = new Float64Array(n);
    this.grad = new Float64Array(n);
    this.s = this.env.reset();
  }

  epsilon(): number {
    const { epsStart, epsEnd, epsDecay } = this.cfg;
    // the lab's schedule: EPS_END + (EPS_START - EPS_END) * 0.99 ** (step / EPS_DECAY * 100)
    return epsEnd + (epsStart - epsEnd) * dexp((this.step / epsDecay) * 100 * LN_099);
  }

  greedy(s: number[]): number {
    const [q0, q1] = this.q.forward(s);
    return q1 > q0 ? 1 : 0;
  }

  /** One environment step + (maybe) one learning step. Returns true when an episode ended. */
  tick(): boolean {
    const eps = this.epsilon();
    this.lastExplore = this.rand() < eps;
    const a = this.lastExplore ? (this.rand() < 0.5 ? 0 : 1) : this.greedy(this.s);
    this.lastAction = a;
    const { s: s2, r, terminated, truncated } = this.env.step(a);
    this.last = { s2, terminated, truncated };
    // truncated (time limit) is not failure: don't cut bootstrapping
    this.push({ s: this.s, a, r, s2, done: terminated ? 1 : 0 });
    this.s = s2;
    this.epReward += r;
    this.step += 1;
    if (this.buffer.length >= this.cfg.learnStart) this.learn();
    if (this.step % this.cfg.targetSync === 0) {
      this.target.copyFrom(this.q);
      this.syncs += 1;
    }
    if (terminated || truncated) {
      this.episodeRewards.push(this.epReward);
      this.episode += 1;
      this.epReward = 0;
      this.s = this.env.reset();
      return true;
    }
    return false;
  }

  private push(tr: Transition) {
    if (this.buffer.length < this.cfg.bufferSize) this.buffer.push(tr);
    else this.buffer[this.bufIdx] = tr;
    this.bufIdx = (this.bufIdx + 1) % this.cfg.bufferSize;
  }

  private learn() {
    const { batch, gamma, lr } = this.cfg;
    const grad = this.grad;
    grad.fill(0);
    const acts = { x: [] as number[], a1: [] as number[], a2: [] as number[] };
    const n = this.buffer.length;
    let loss = 0;
    this.lastBatch = [];
    for (let i = 0; i < batch; i++) {
      const idx = Math.floor(this.rand() * n);
      this.lastBatch.push(idx);
      const tr = this.buffer[idx];
      const qsa = this.q.forward(tr.s, acts)[tr.a];
      // TD target with the target network (stops self-chasing)
      const [t0, t1] = this.target.forward(tr.s2);
      let next = Math.max(t0, t1);
      if (this.cfg.double) {
        const [o0, o1] = this.q.forward(tr.s2);
        next = o1 > o0 ? t1 : t0;
      }
      const target = tr.r + gamma * next * (1 - tr.done);
      const diff = qsa - target;
      // Huber (smooth L1) loss
      const ad = Math.abs(diff);
      loss += ad < 1 ? 0.5 * diff * diff : ad - 0.5;
      const g = (ad < 1 ? diff : Math.sign(diff)) / batch;
      this.q.backward(acts, tr.a, g, grad);
    }
    this.lastLoss = loss / batch;
    // clip global grad norm to 10
    let norm = 0;
    for (let i = 0; i < grad.length; i++) norm += grad[i] * grad[i];
    norm = Math.sqrt(norm);
    const clip = norm > 10 ? 10 / norm : 1;
    // Adam
    const b1 = 0.9;
    const b2 = 0.999;
    this.b1t *= b1;
    this.b2t *= b2;
    const c1 = 1 - this.b1t;
    const c2 = 1 - this.b2t;
    const p = this.q.p;
    for (let i = 0; i < p.length; i++) {
      const gi = grad[i] * clip;
      this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
      this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
      p[i] -= (lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + 1e-8);
    }
  }

  recentMean(k = 20): number {
    const r = this.episodeRewards.slice(-k);
    return r.length ? r.reduce((a, b) => a + b, 0) / r.length : 0;
  }
}

/** Greedy test (ε = 0, no learning) on a separate environment; returns each episode's length. */
export function evaluate(net: QNet, episodes: number, seed: number): number[] {
  const env = new CartPole(mulberry(seed));
  const out: number[] = [];
  for (let e = 0; e < episodes; e++) {
    env.reset();
    for (;;) {
      const [q0, q1] = net.forward(env.s);
      const { terminated, truncated } = env.step(q1 > q0 ? 1 : 0);
      if (terminated || truncated) break;
    }
    out.push(env.t);
  }
  return out;
}
