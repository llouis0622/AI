/**
 * A tiny multi-layer perceptron for in-browser demos (2 inputs → hidden layers → 1 sigmoid output),
 * trained with mini-batch Adam on binary cross-entropy. Small enough to retrain every frame.
 */
import { rng } from '../core/math';

export type Act = 'tanh' | 'relu';

export interface Net {
  sizes: number[];
  act: Act;
  W: Float64Array[]; // W[l][o * in + i], l = 0 .. L-1
  b: Float64Array[];
  mW: Float64Array[];
  vW: Float64Array[];
  mb: Float64Array[];
  vb: Float64Array[];
  t: number;
}

export function createNet(sizes: number[], act: Act, seed = 1): Net {
  const r = rng(seed);
  const W: Float64Array[] = [];
  const b: Float64Array[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l];
    const out = sizes[l + 1];
    const std = act === 'relu' ? Math.sqrt(2 / fanIn) : Math.sqrt(1 / fanIn);
    const w = new Float64Array(out * fanIn);
    for (let k = 0; k < w.length; k++) w[k] = r.normal() * std;
    W.push(w);
    // small positive bias keeps ReLU units alive at the start
    b.push(new Float64Array(out).fill(act === 'relu' ? 0.05 : 0));
  }
  const zeros = (arr: Float64Array[]) => arr.map((a) => new Float64Array(a.length));
  return { sizes, act, W, b, mW: zeros(W), vW: zeros(W), mb: zeros(b), vb: zeros(b), t: 0 };
}

const actF = (act: Act, z: number) => (act === 'relu' ? (z > 0 ? z : 0) : Math.tanh(z));
const actD = (act: Act, z: number, a: number) => (act === 'relu' ? (z > 0 ? 1 : 0) : 1 - a * a);

/** Forward pass. Returns the output probability; fills `acts` (per layer activations) if given. */
export function forward(net: Net, x1: number, x2: number, acts?: Float64Array[], zs?: Float64Array[]): number {
  const L = net.sizes.length - 1;
  let prev: Float64Array = acts?.[0] ?? new Float64Array(2);
  prev[0] = x1;
  prev[1] = x2;
  let out = 0;
  for (let l = 0; l < L; l++) {
    const inN = net.sizes[l];
    const outN = net.sizes[l + 1];
    const w = net.W[l];
    const bb = net.b[l];
    const cur: Float64Array = acts?.[l + 1] ?? new Float64Array(outN);
    const zz = zs?.[l + 1];
    for (let o = 0; o < outN; o++) {
      let s = bb[o];
      const row = o * inN;
      for (let i = 0; i < inN; i++) s += w[row + i] * prev[i];
      if (zz) zz[o] = s;
      if (l === L - 1) {
        cur[o] = 1 / (1 + Math.exp(-s));
        out = cur[o];
      } else cur[o] = actF(net.act, s);
    }
    prev = cur;
  }
  return out;
}

export function makeBuffers(net: Net) {
  return {
    acts: net.sizes.map((n) => new Float64Array(n)),
    zs: net.sizes.map((n) => new Float64Array(n))
  };
}

/** One Adam step on a mini-batch; returns mean BCE loss of the batch. */
export function trainBatch(net: Net, X: [number, number][], Y: number[], idx: number[], lr: number): number {
  const L = net.sizes.length - 1;
  const gW = net.W.map((w) => new Float64Array(w.length));
  const gb = net.b.map((b) => new Float64Array(b.length));
  const { acts, zs } = makeBuffers(net);
  const deltas = net.sizes.map((n) => new Float64Array(n));
  let loss = 0;
  for (const k of idx) {
    const p = forward(net, X[k][0], X[k][1], acts, zs);
    const y = Y[k];
    const pc = Math.min(1 - 1e-7, Math.max(1e-7, p));
    loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
    deltas[L][0] = p - y;
    for (let l = L - 1; l >= 0; l--) {
      const inN = net.sizes[l];
      const outN = net.sizes[l + 1];
      const w = net.W[l];
      const dOut = deltas[l + 1];
      const aIn = acts[l];
      const gw = gW[l];
      const gbl = gb[l];
      for (let o = 0; o < outN; o++) {
        const d = dOut[o];
        gbl[o] += d;
        const row = o * inN;
        for (let i = 0; i < inN; i++) gw[row + i] += d * aIn[i];
      }
      if (l > 0) {
        const dIn = deltas[l];
        for (let i = 0; i < inN; i++) {
          let s = 0;
          for (let o = 0; o < outN; o++) s += w[o * inN + i] * dOut[o];
          dIn[i] = s * actD(net.act, zs[l][i], acts[l][i]);
        }
      }
    }
  }
  const n = idx.length;
  net.t += 1;
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;
  const c1 = 1 - b1 ** net.t;
  const c2 = 1 - b2 ** net.t;
  const step = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
    for (let k = 0; k < p.length; k++) {
      const gk = g[k] / n;
      m[k] = b1 * m[k] + (1 - b1) * gk;
      v[k] = b2 * v[k] + (1 - b2) * gk * gk;
      p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + eps);
    }
  };
  for (let l = 0; l < L; l++) {
    step(net.W[l], gW[l], net.mW[l], net.vW[l]);
    step(net.b[l], gb[l], net.mb[l], net.vb[l]);
  }
  return loss / n;
}

export function evaluate(net: Net, X: [number, number][], Y: number[]) {
  let loss = 0;
  let correct = 0;
  for (let k = 0; k < X.length; k++) {
    const p = forward(net, X[k][0], X[k][1]);
    const pc = Math.min(1 - 1e-7, Math.max(1e-7, p));
    loss += -(Y[k] * Math.log(pc) + (1 - Y[k]) * Math.log(1 - pc));
    if ((p >= 0.5 ? 1 : 0) === Y[k]) correct++;
  }
  return { loss: loss / X.length, acc: correct / X.length };
}

/* ---------- Toy 2D datasets in [-1, 1]² ---------- */

export type DatasetId = 'circle' | 'xor' | 'spiral' | 'blobs';

export function makeDataset(id: DatasetId, n: number, noise: number, seed: number) {
  const r = rng(seed);
  const X: [number, number][] = [];
  const Y: number[] = [];
  const push = (x: number, y: number, label: number) => {
    X.push([Math.max(-1, Math.min(1, x)), Math.max(-1, Math.min(1, y))]);
    Y.push(label);
  };
  for (let k = 0; k < n; k++) {
    const label = k % 2;
    const jitter = () => r.normal() * noise * 0.18;
    if (id === 'circle') {
      const radius = label === 1 ? r.next() * 0.42 : 0.62 + r.next() * 0.33;
      const a = r.next() * Math.PI * 2;
      push(radius * Math.cos(a) + jitter(), radius * Math.sin(a) + jitter(), label);
    } else if (id === 'xor') {
      let x = r.next() * 2 - 1;
      let y = r.next() * 2 - 1;
      const pad = 0.08;
      x += x > 0 ? pad : -pad;
      y += y > 0 ? pad : -pad;
      const l = x * y > 0 ? 1 : 0;
      push(x * 0.9 + jitter(), y * 0.9 + jitter(), l);
    } else if (id === 'spiral') {
      const i = Math.floor(k / 2);
      const m = Math.ceil(n / 2);
      const t = i / m;
      const rad = 0.08 + t * 0.86;
      const ang = t * 1.75 * Math.PI * 2 + (label === 1 ? Math.PI : 0);
      push(rad * Math.cos(ang) + jitter(), rad * Math.sin(ang) + jitter(), label);
    } else {
      const cx = label === 1 ? 0.45 : -0.45;
      push(cx + r.normal() * (0.2 + noise * 0.25), cx + r.normal() * (0.2 + noise * 0.25), label);
    }
  }
  return { X, Y };
}
