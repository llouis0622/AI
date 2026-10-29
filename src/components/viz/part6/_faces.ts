/**
 * A toy "decoder" for chapter 19: turns a 2D latent vector z into face parameters.
 * The VAE decoder is smooth everywhere (its latent space is packed with meaning);
 * the plain autoencoder only knows the exact spots where its training photos landed.
 */
import { sigmoid } from '../core/math';

export type Z = [number, number];

export interface FaceParams {
  /** -1 화남 … 1 웃음 */
  mood: number;
  /** 0 갸름 … 1 동그람 */
  round: number;
  /** 0 … 1 eye size */
  eye: number;
  /** hair hue in degrees */
  hue: number;
  /** 0 … 1 hair volume */
  hair: number;
  /** 0 = clean picture, 1 = total garbage */
  glitch: number;
  z: Z;
}

/** Where the 8 training photos sit: `z` for the VAE (overlapping clouds), `ae` for the plain autoencoder. */
export const TRAIN: { z: Z; ae: Z }[] = [
  { z: [1.35, 1.15], ae: [0.87, 0.87] },
  { z: [-1.45, 1.05], ae: [-2.6, 1.73] },
  { z: [1.2, -1.35], ae: [2.6, -0.87] },
  { z: [-1.25, -1.3], ae: [-0.87, -2.6] },
  { z: [0.05, 1.85], ae: [1.73, 2.6] },
  { z: [0.15, -1.9], ae: [1.73, -2.6] },
  { z: [1.9, 0.05], ae: [-2.6, -1.73] },
  { z: [-1.9, -0.05], ae: [-0.87, 1.73] }
];

export const CLOUD_SIGMA = 0.55;
/** Below this glitch level a face still counts as "멀쩡한 얼굴". */
export const CLEAN = 0.3;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function hash(i: number, j: number, s: number) {
  let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(s, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return ((h >>> 0) / 4294967295) * 2 - 1;
}

/** Smooth 2D value noise in [-1, 1]. */
export function vnoise(x: number, y: number, s: number) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi, s);
  const b = hash(xi + 1, yi, s);
  const c = hash(xi, yi + 1, s);
  const d = hash(xi + 1, yi + 1, s);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/** The well-trained VAE decoder: every z gives a plausible face (only far corners get shaky). */
export function decodeVae(z: Z): FaceParams {
  const [a, b] = z;
  return {
    mood: Math.tanh(0.85 * a),
    round: sigmoid(1.25 * b),
    eye: sigmoid(0.9 * b + 0.35 * a),
    hue: 200 + 150 * sigmoid(0.9 * (a - b)),
    hair: sigmoid(0.9 * (a + b)),
    glitch: 0.7 * smooth(3.3, 4.6, Math.hypot(a, b)),
    z
  };
}

/** The plain autoencoder: perfect on its training spots, garbage in the empty space between them. */
export function decodeAe(z: Z): FaceParams {
  let best = 0;
  let bestD = Infinity;
  TRAIN.forEach((t, i) => {
    const d = Math.hypot(z[0] - t.ae[0], z[1] - t.ae[1]);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  const conf = Math.exp(-(bestD * bestD) / (2 * 0.4 * 0.4));
  const face = decodeVae(TRAIN[best].z);
  return { ...face, glitch: 1 - smooth(0.15, 0.85, conf), z };
}
